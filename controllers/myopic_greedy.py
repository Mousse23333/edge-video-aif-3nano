"""
Myopic Greedy Controller.

At each step, enumerates all possible actions and picks the one predicted
to yield the best immediate outcome (one-step lookahead only, no belief).

Uses a static profiling table built from benchmark data.
No learning, no future planning, no uncertainty modeling.

Key difference from Heuristic: heuristic uses fixed rules;
Myopic enumerates all actions and scores them quantitatively.
Key difference from AIF: no belief update, no epistemic value,
no multi-step planning.
"""

import yaml
import numpy as np
from engine.episode import ControllerInterface


# ── Profiling table (from benchmark_imgsz.py results on Orin) ───────────────
# Maps (mode, n_inference_streams) -> (expected_fps, expected_p95_ms)
# n_inference_streams = number of streams doing inference (FULL or LITE)

PROFILING_TABLE = {
    # (mode, n_infer): (fps_per_stream, p95_latency_ms)
    ("FULL", 1):  (30.0,  35.0),
    ("FULL", 2):  (21.6,  47.6),
    ("FULL", 3):  (16.7,  59.4),
    ("FULL", 4):  (13.0,  74.6),
    ("FULL", 5):  (10.6,  91.5),
    ("FULL", 6):  ( 8.7, 107.3),
    ("FULL", 7):  ( 7.9, 112.5),
    ("FULL", 8):  ( 7.2, 129.6),
    # LITE imgsz=320
    ("LITE", 1):  (30.8,  30.7),
    ("LITE", 2):  (25.5,  35.7),
    ("LITE", 3):  (21.2,  40.7),
    ("LITE", 4):  (17.6,  47.4),
    ("LITE", 5):  (14.7,  57.0),
    ("LITE", 6):  (12.4,  66.3),
    ("LITE", 7):  (10.8,  74.2),
    ("LITE", 8):  ( 9.4,  86.6),
    ("LITE", 9):  ( 8.5,  95.0),
    ("LITE", 10): ( 7.7, 103.0),
    # SKIP: no inference, latency irrelevant
    ("SKIP", 0):  (999.0,  0.0),
}


def predict_stream_perf(mode, n_infer_streams):
    """Predict per-stream FPS and P95 latency given mode and total inference streams."""
    if mode == "SKIP":
        return 999.0, 0.0
    if mode == "OFFLOAD":
        # Remote inference on Nano: ~7 fps, ~140ms round-trip
        # Independent of local GPU load
        return 7.0, 140.0
    key = (mode, n_infer_streams)
    if key in PROFILING_TABLE:
        return PROFILING_TABLE[key]
    # Extrapolate: beyond table, use trend
    max_n = max(k[1] for k in PROFILING_TABLE if k[0] == mode)
    fps_max, p95_max = PROFILING_TABLE[(mode, max_n)]
    extra = n_infer_streams - max_n
    fps = max(1.0, fps_max - extra * 1.0)
    p95 = min(500.0, p95_max + extra * 15.0)
    return fps, p95


class MyopicGreedyController(ControllerInterface):
    """
    Scores each action by predicting next-step SLO satisfaction.
    Picks the action with highest score. No learning, no look-ahead.
    """

    def __init__(self, config_dir="/app/config"):
        with open(f"{config_dir}/slo.yaml") as f:
            slo = yaml.safe_load(f)
        with open(f"{config_dir}/switch_cost.yaml") as f:
            self.switch_cfg = yaml.safe_load(f)
        with open(f"{config_dir}/action_space.yaml") as f:
            action_cfg = yaml.safe_load(f)

        hc = slo["hard_constraints"]
        self.lat_thresh = hc["p95_latency_ms"]["threshold"]    # 150
        self.fps_thresh = hc["min_fps"]["threshold"]           # 10
        self.max_skip_ratio = hc["max_skip_ratio"]["threshold"] # 0.3
        self.min_dwell = self.switch_cfg.get("min_dwell_before_exit", {})

        # OFFLOAD config
        offload_cfg = action_cfg.get("modes", {}).get("OFFLOAD", {})
        self.offload_enabled = offload_cfg.get("enabled", False)
        offload_urls = offload_cfg.get("urls", [])
        self.max_offload = len(offload_urls) if self.offload_enabled else 0

        self.modes = ["FULL", "LITE", "SKIP"]
        if self.offload_enabled:
            self.modes.append("OFFLOAD")

    def on_episode_start(self, config):
        pass

    def select_action(self, obs, stream_manager):
        per_stream = obs["per_stream"]
        n_active = obs["global"]["n_active_streams"]
        if n_active == 0:
            return None

        stream_ids = list(per_stream.keys())

        # Current modes
        current_modes = {sid: per_stream[sid]["current_mode"] for sid in stream_ids}

        # Count current inference streams (FULL + LITE)
        n_infer_current = sum(1 for m in current_modes.values() if m in ("FULL", "LITE"))

        # Evaluate NO-OP first
        best_score = self._score_config(current_modes, n_infer_current)
        best_action = None

        # Enumerate all possible single-stream mode changes
        for sid in stream_ids:
            old_mode = current_modes[sid]

            # Check min_dwell
            dwell = stream_manager.get_mode_dwell(sid)
            min_d = self.min_dwell.get(old_mode, 1)
            if dwell < min_d:
                continue

            for new_mode in self.modes:
                if new_mode == old_mode:
                    continue

                # Check skip ratio constraint
                if new_mode == "SKIP":
                    n_skip = sum(1 for m in current_modes.values() if m == "SKIP")
                    if (n_skip + 1) / n_active > self.max_skip_ratio:
                        continue

                # Check offload capacity
                if new_mode == "OFFLOAD":
                    n_offload = sum(1 for m in current_modes.values()
                                    if m == "OFFLOAD")
                    if n_offload >= self.max_offload:
                        continue

                # Simulate the config after this action
                simulated = dict(current_modes)
                simulated[sid] = new_mode
                n_infer_sim = sum(1 for m in simulated.values() if m in ("FULL", "LITE"))

                # Get switch penalty
                penalty = self._switch_penalty(old_mode, new_mode)

                score = self._score_config(simulated, n_infer_sim) - penalty

                if score > best_score:
                    best_score = score
                    best_action = (sid, new_mode)

        return best_action

    def _score_config(self, mode_map, n_infer):
        """Score a mode configuration based on predicted SLO satisfaction."""
        total_score = 0.0

        for sid, mode in mode_map.items():
            if mode == "SKIP":
                # SKIP: penalize for missing detections, but no SLO violation
                total_score += 0.3  # partial credit (graceful, not zero)
                continue

            if mode == "OFFLOAD":
                # OFFLOAD: still gets detections but at low fps (~7)
                # Better than SKIP (has detections), worse than local inference
                fps, p95 = predict_stream_perf("OFFLOAD", 0)
                fps_ok = fps >= self.fps_thresh
                lat_ok = p95 <= self.lat_thresh
                if fps_ok and lat_ok:
                    total_score += 0.9  # good but remote
                else:
                    total_score += 0.5  # detections but SLO violated
                continue

            fps, p95 = predict_stream_perf(mode, n_infer)

            fps_ok = fps >= self.fps_thresh
            lat_ok = p95 <= self.lat_thresh

            if fps_ok and lat_ok:
                stream_score = 1.0
                # Bonus for headroom
                fps_margin = (fps - self.fps_thresh) / self.fps_thresh
                lat_margin = (self.lat_thresh - p95) / self.lat_thresh
                stream_score += 0.2 * min(fps_margin, 0.5) + 0.2 * min(lat_margin, 0.5)
            else:
                # Penalize proportionally to violation severity
                stream_score = 0.0
                if not fps_ok:
                    stream_score -= (self.fps_thresh - fps) / self.fps_thresh
                if not lat_ok:
                    stream_score -= (p95 - self.lat_thresh) / self.lat_thresh

            total_score += stream_score

        return total_score

    def _switch_penalty(self, from_mode, to_mode):
        """Look up switch penalty score (0=none, 4=high)."""
        key = f"{from_mode}->{to_mode}"
        t = self.switch_cfg.get("transitions", {}).get(key, {})
        score = t.get("penalty_score", 1)
        return score * 0.05  # scale down so it doesn't overwhelm SLO score
