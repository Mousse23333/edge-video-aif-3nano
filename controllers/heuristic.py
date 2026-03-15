"""
Threshold-based heuristic controller.

Layered degradation with hysteresis recovery:
  Overload detected  →  demote worst stream: FULL→LITE→SKIP
  System recovered   →  promote best candidate: SKIP→LITE→FULL

Decisions use only the observation interface (same as AIF/RL).
"""

import yaml
import os
from engine.episode import ControllerInterface


class HeuristicController(ControllerInterface):
    """
    Rule-based controller with directional hysteresis.

    Downgrade: triggered when ANY stream violates SLO.
    Upgrade:   triggered only after ALL streams satisfy SLO for
               `recovery_patience` consecutive windows.
    """

    def __init__(self, config_dir="/app/config"):
        with open(f"{config_dir}/slo.yaml") as f:
            slo = yaml.safe_load(f)
        with open(f"{config_dir}/switch_cost.yaml") as f:
            self.switch_cfg = yaml.safe_load(f)

        hc = slo["hard_constraints"]
        self.lat_thresh = hc["p95_latency_ms"]["threshold"]      # 150
        self.fps_thresh = hc["min_fps"]["threshold"]              # 10
        self.max_skip_ratio = hc["max_skip_ratio"]["threshold"]   # 0.3
        self.max_gpu_util = hc["max_gpu_util"]["threshold"]       # 90

        # Hysteresis: require this many consecutive clean windows before upgrade
        self.recovery_patience = 5
        self._clean_windows = 0

        # Upgrade thresholds (more conservative than SLO limits)
        self.upgrade_lat_thresh = self.lat_thresh * 0.5   # 75ms
        self.upgrade_fps_thresh = self.fps_thresh * 1.3   # 13
        self.upgrade_gpu_thresh = self.max_gpu_util * 0.65  # 58.5

        self.min_dwell = self.switch_cfg.get("min_dwell_before_exit", {})

        # OFFLOAD config
        with open(f"{config_dir}/action_space.yaml") as f:
            action_cfg = yaml.safe_load(f)
        offload_cfg = action_cfg.get("modes", {}).get("OFFLOAD", {})
        self.offload_enabled = offload_cfg.get("enabled", False)
        offload_urls = offload_cfg.get("urls", [])
        self.max_offload = len(offload_urls) if self.offload_enabled else 0

        # Mode hierarchy for demotion/promotion
        # FULL → LITE → OFFLOAD (if slots) → SKIP
        self.mode_rank = {"FULL": 3, "LITE": 2, "OFFLOAD": 1, "SKIP": 0}
        if self.offload_enabled:
            self.demote_map = {"FULL": "LITE", "LITE": "OFFLOAD", "OFFLOAD": "SKIP"}
            self.promote_map = {"SKIP": "OFFLOAD", "OFFLOAD": "LITE", "LITE": "FULL"}
        else:
            self.demote_map = {"FULL": "LITE", "LITE": "SKIP"}
            self.promote_map = {"SKIP": "LITE", "LITE": "FULL"}

    def on_episode_start(self, config):
        self._clean_windows = 0

    def select_action(self, obs, stream_manager):
        per_stream = obs["per_stream"]
        gl = obs["global"]
        gpu = gl.get("gpu_util_avg", 0)
        n_active = gl.get("n_active_streams", 0)

        if n_active == 0:
            return None

        # --- Identify violating and healthy streams ---
        violating = []    # (stream_id, severity_score)
        all_healthy = True

        for sid, ps in per_stream.items():
            mode = ps.get("current_mode", "FULL")
            if mode == "SKIP":
                continue  # SKIP streams are exempt from SLO

            # Use inference-specific metrics for accurate SLO checking
            lat = ps.get("infer_latency_p95_ms", 0) or ps.get("latency_p95_ms", 0)
            fps = ps.get("infer_fps", 0) or ps.get("fps_avg", 999)

            lat_violated = lat > self.lat_thresh
            fps_violated = fps < self.fps_thresh

            if lat_violated or fps_violated:
                all_healthy = False
                # Severity: how badly is it violating?
                severity = 0
                if lat_violated:
                    severity += (lat - self.lat_thresh) / self.lat_thresh
                if fps_violated:
                    severity += (self.fps_thresh - fps) / self.fps_thresh
                violating.append((sid, severity))

        # --- Check GPU pressure ---
        gpu_pressure = gpu > self.max_gpu_util

        # --- Downgrade path ---
        if violating or gpu_pressure:
            self._clean_windows = 0
            return self._try_demote(violating, per_stream, stream_manager)

        # --- Upgrade path (with hysteresis) ---
        if all_healthy:
            self._clean_windows += 1
        else:
            self._clean_windows = 0

        if self._clean_windows >= self.recovery_patience:
            return self._try_promote(per_stream, gl, stream_manager)

        # Special case: if there are SKIP streams AND the inference streams
        # are violating, promoting SKIP→LITE might actually help by improving
        # batch efficiency (more streams = better GPU utilization up to limit)
        n_skip = sum(1 for ps in per_stream.values()
                     if ps.get("current_mode") == "SKIP")
        n_lite = sum(1 for ps in per_stream.values()
                     if ps.get("current_mode") == "LITE")
        if n_skip > 0 and n_lite <= 5 and violating:
            return self._try_promote(per_stream, gl, stream_manager)

        return None  # NO-OP

    def _try_demote(self, violating, per_stream, sm):
        """Demote the worst-performing stream that can be demoted."""

        # Sort by severity (worst first)
        if violating:
            candidates = sorted(violating, key=lambda x: -x[1])
        else:
            # GPU pressure but no per-stream violation:
            # demote the stream with highest latency among FULL streams
            full_streams = [
                (sid, ps.get("latency_p95_ms", 0))
                for sid, ps in per_stream.items()
                if ps.get("current_mode") == "FULL"
            ]
            if not full_streams:
                return None
            candidates = sorted(full_streams, key=lambda x: -x[1])

        for sid, _ in candidates:
            mode = sm.get_mode(sid)
            if mode is None:
                continue

            new_mode = self.demote_map.get(mode)
            if new_mode is None:
                continue  # already at SKIP

            # Check skip ratio constraint before demoting to SKIP
            if new_mode == "SKIP":
                if not self._can_add_skip(per_stream):
                    continue

            # Check offload capacity before demoting to OFFLOAD
            if new_mode == "OFFLOAD":
                n_offload = sum(1 for ps in per_stream.values()
                                if ps.get("current_mode") == "OFFLOAD")
                if n_offload >= self.max_offload:
                    # Skip OFFLOAD, try next demotion (SKIP)
                    new_mode = "SKIP"
                    if not self._can_add_skip(per_stream):
                        continue

            # Check min_dwell
            dwell = sm.get_mode_dwell(sid)
            min_d = self.min_dwell.get(mode, 1)
            if dwell < min_d:
                continue

            return (sid, new_mode)

        return None

    def _try_promote(self, per_stream, gl, sm):
        """Promote the best candidate stream to a higher mode."""
        gpu = gl.get("gpu_util_avg", 0)

        # Don't promote if GPU is already getting warm
        if gpu > self.upgrade_gpu_thresh:
            return None

        # Find promotable streams, prioritize SKIP→LITE over LITE→FULL
        candidates = []
        for sid, ps in per_stream.items():
            mode = ps.get("current_mode")
            new_mode = self.promote_map.get(mode)
            if new_mode is None:
                continue  # already FULL

            # Check dwell
            dwell = sm.get_mode_dwell(sid)
            min_d = self.min_dwell.get(mode, 1)
            if dwell < min_d:
                continue

            # Priority: lower mode rank = promote first (SKIP before LITE)
            candidates.append((sid, self.mode_rank.get(mode, 0), new_mode))

        if not candidates:
            return None

        # Promote lowest-ranked stream first
        candidates.sort(key=lambda x: x[1])
        sid, _, new_mode = candidates[0]
        return (sid, new_mode)

    def _can_add_skip(self, per_stream):
        """Check if adding one more SKIP stream would exceed max_skip_ratio."""
        n_total = len(per_stream)
        if n_total == 0:
            return False
        n_skip = sum(1 for ps in per_stream.values()
                     if ps.get("current_mode") == "SKIP")
        return (n_skip + 1) / n_total <= self.max_skip_ratio
