"""
Tabular Q-Learning Controller.

Uses the same 3-level state discretization as AIF (LOW/MEDIUM/HIGH load)
for direct comparability. Very sample-efficient: Q-table converges in ~20 steps.

State: (load_level, n_active_streams_binned)
Action: (stream_id, new_mode) or NO-OP

Key difference from AIF: maximizes scalar reward, no belief/generative model.
Key difference from DQN: tabular (no neural net), directly comparable state space.
"""

import yaml
import numpy as np
import random
from collections import defaultdict
from engine.episode import ControllerInterface

# Load states
LOAD_LOW    = 0
LOAD_MEDIUM = 1
LOAD_HIGH   = 2

# State discretization (mirrors AIF)
LOAD_LOW    = 0   # avg infer_fps >= 15
LOAD_MEDIUM = 1   # avg infer_fps in [10, 15)
LOAD_HIGH   = 2   # avg infer_fps < 10

MODES = ["FULL", "LITE", "SKIP"]


def discretize_state(obs):
    """Map observation to discrete (load_level, n_streams_bin)."""
    per_stream = obs["per_stream"]
    fps_vals = [
        ps.get("infer_fps", ps.get("fps_avg", 0))
        for ps in per_stream.values()
        if ps.get("current_mode") not in ("SKIP",)
    ]
    avg_fps = np.mean(fps_vals) if fps_vals else 0

    if avg_fps >= 15:
        load = LOAD_LOW
    elif avg_fps >= 10:
        load = LOAD_MEDIUM
    else:
        load = LOAD_HIGH

    n = obs["global"]["n_active_streams"]
    n_bin = min(n // 2, 4)  # bins: 0-1, 2-3, 4-5, 6-7, 8+

    # Mode distribution
    modes = tuple(sorted(
        ps.get("current_mode", "FULL")
        for ps in per_stream.values()
    ))
    mode_sig = sum(1 for m in modes if m == "FULL") * 100 + \
               sum(1 for m in modes if m == "LITE") * 10 + \
               sum(1 for m in modes if m == "SKIP")

    return (load, n_bin, min(mode_sig, 999))


class TabularQLearningController(ControllerInterface):
    """
    Online tabular Q-learning. State is discretized into load level + stream count.
    Learns a Q-value for each (state, action_type) pair during the episode.

    Actions are abstracted into 5 types to keep Q-table small:
      0: NO-OP
      1: demote worst FULL -> LITE
      2: demote worst LITE -> SKIP
      3: promote best SKIP -> LITE
      4: promote best LITE -> FULL
    """

    ACTION_DEMOTE_FULL    = 1
    ACTION_DEMOTE_LITE    = 2
    ACTION_PROMOTE_SKIP   = 3
    ACTION_PROMOTE_LITE   = 4
    ACTION_OFFLOAD        = 5   # FULL/LITE → OFFLOAD
    ACTION_RECALL_OFFLOAD = 6   # OFFLOAD → LITE
    N_ACTION_TYPES = 7

    def __init__(self, config_dir="/app/config",
                 lr=0.3, gamma=0.9,
                 epsilon_start=0.3, epsilon_end=0.01):
        with open(f"{config_dir}/slo.yaml") as f:
            slo = yaml.safe_load(f)
        with open(f"{config_dir}/switch_cost.yaml") as f:
            self.switch_cfg = yaml.safe_load(f)

        hc = slo["hard_constraints"]
        self.lat_thresh = hc["p95_latency_ms"]["threshold"]
        self.fps_thresh = hc["min_fps"]["threshold"]
        self.max_skip   = hc["max_skip_ratio"]["threshold"]
        self.min_dwell  = self.switch_cfg.get("min_dwell_before_exit", {})

        rl_r = slo["rl_reward"]
        self.r_slo_ok   = rl_r["slo_satisfied_bonus"]
        self.r_lat_viol = rl_r["latency_violation_penalty"]
        self.r_fps_viol = rl_r["fps_violation_penalty"]
        self.r_skip     = rl_r["skip_penalty"]
        self.r_offload  = rl_r.get("offload_penalty", -0.2)
        self.r_noop     = rl_r["no_op_bonus"]

        # OFFLOAD config
        with open(f"{config_dir}/action_space.yaml") as f:
            action_cfg = yaml.safe_load(f)
        offload_cfg = action_cfg.get("modes", {}).get("OFFLOAD", {})
        self.offload_enabled = offload_cfg.get("enabled", False)
        offload_urls = offload_cfg.get("urls", [])
        self.max_offload = len(offload_urls) if self.offload_enabled else 0

        self.lr      = lr
        self.gamma   = gamma
        self.epsilon = epsilon_start
        self.eps_end = epsilon_end

        # Q-table: defaultdict so unseen states start at 0
        # PERSISTS across episodes for cross-episode learning
        self.Q = defaultdict(lambda: np.zeros(self.N_ACTION_TYPES))

        self._prev_state  = None
        self._prev_action = None
        self._step = 0
        self._total_steps = 0
        self._episode = 0
        self._eval_mode = False

    def set_eval_mode(self, eval_mode=True):
        """Switch between training (exploration) and evaluation (exploitation)."""
        self._eval_mode = eval_mode

    def on_episode_start(self, config):
        # Reset per-episode state only; Q-table persists across episodes
        self._prev_state  = None
        self._prev_action = None
        self._step = 0
        self._episode += 1

    def select_action(self, obs, stream_manager):
        per_stream = obs["per_stream"]
        n_active   = obs["global"]["n_active_streams"]
        if n_active == 0:
            return None

        state = discretize_state(obs)

        # Update Q from previous transition
        if self._prev_state is not None:
            reward = self._compute_reward(obs)
            q_next = np.max(self.Q[state])
            td_target = reward + self.gamma * q_next
            td_error  = td_target - self.Q[self._prev_state][self._prev_action]
            self.Q[self._prev_state][self._prev_action] += self.lr * td_error

        # Epsilon: eval mode uses near-greedy, training decays across total steps
        if self._eval_mode:
            eps = self.eps_end
        else:
            eps = max(self.eps_end,
                      self.epsilon - self._total_steps * (self.epsilon - self.eps_end) / 300)

        # Get valid abstract action types
        valid = self._valid_action_types(per_stream, stream_manager, n_active)
        if not valid:
            return None

        # Epsilon-greedy
        if random.random() < eps:
            action_type = random.choice(valid)
        else:
            q_vals = self.Q[state].copy()
            mask = np.full(self.N_ACTION_TYPES, -1e9)
            for a in valid:
                mask[a] = q_vals[a]
            action_type = int(np.argmax(mask))

        self._prev_state  = state
        self._prev_action = action_type
        self._step += 1
        self._total_steps += 1

        return self._resolve_action(action_type, per_stream, stream_manager, n_active)

    def _valid_action_types(self, per_stream, sm, n_active):
        valid = [0]  # NO-OP always valid
        modes = {sid: ps.get("current_mode", "FULL")
                 for sid, ps in per_stream.items()}

        n_skip = sum(1 for m in modes.values() if m == "SKIP")
        n_offload = sum(1 for m in modes.values() if m == "OFFLOAD")

        for sid, mode in modes.items():
            dwell = sm.get_mode_dwell(sid)
            min_d = self.min_dwell.get(mode, 1)
            if dwell < min_d:
                continue
            if mode == "FULL":
                valid.append(self.ACTION_DEMOTE_FULL)
                if self.offload_enabled and n_offload < self.max_offload:
                    valid.append(self.ACTION_OFFLOAD)
            elif mode == "LITE":
                if (n_skip + 1) / n_active <= self.max_skip:
                    valid.append(self.ACTION_DEMOTE_LITE)
                valid.append(self.ACTION_PROMOTE_LITE)
                if self.offload_enabled and n_offload < self.max_offload:
                    valid.append(self.ACTION_OFFLOAD)
            elif mode == "SKIP":
                valid.append(self.ACTION_PROMOTE_SKIP)
            elif mode == "OFFLOAD":
                valid.append(self.ACTION_RECALL_OFFLOAD)

        return list(set(valid))

    def _resolve_action(self, action_type, per_stream, sm, n_active):
        """Convert abstract action type to concrete (stream_id, new_mode)."""
        if action_type == 0:
            return None

        modes = {sid: ps.get("current_mode", "FULL")
                 for sid, ps in per_stream.items()}
        fps   = {sid: ps.get("infer_fps", ps.get("fps_avg", 0))
                 for sid, ps in per_stream.items()}

        if action_type == self.ACTION_DEMOTE_FULL:
            # Demote FULL stream with lowest FPS
            candidates = [(sid, fps[sid]) for sid, m in modes.items()
                          if m == "FULL" and sm.get_mode_dwell(sid) >= self.min_dwell.get("FULL", 1)]
            if not candidates:
                return None
            sid = min(candidates, key=lambda x: x[1])[0]
            return (sid, "LITE")

        elif action_type == self.ACTION_DEMOTE_LITE:
            n_skip = sum(1 for m in modes.values() if m == "SKIP")
            candidates = [(sid, fps[sid]) for sid, m in modes.items()
                          if m == "LITE" and sm.get_mode_dwell(sid) >= self.min_dwell.get("LITE", 1)]
            if not candidates:
                return None
            if (n_skip + 1) / n_active > self.max_skip:
                return None
            sid = min(candidates, key=lambda x: x[1])[0]
            return (sid, "SKIP")

        elif action_type == self.ACTION_PROMOTE_SKIP:
            candidates = [(sid,) for sid, m in modes.items()
                          if m == "SKIP" and sm.get_mode_dwell(sid) >= self.min_dwell.get("SKIP", 2)]
            if not candidates:
                return None
            sid = candidates[0][0]
            return (sid, "LITE")

        elif action_type == self.ACTION_PROMOTE_LITE:
            candidates = [(sid,) for sid, m in modes.items()
                          if m == "LITE" and sm.get_mode_dwell(sid) >= self.min_dwell.get("LITE", 1)]
            if not candidates:
                return None
            sid = candidates[0][0]
            return (sid, "FULL")

        elif action_type == self.ACTION_OFFLOAD:
            # Offload the FULL/LITE stream with lowest FPS
            candidates = [(sid, fps[sid]) for sid, m in modes.items()
                          if m in ("FULL", "LITE")
                          and sm.get_mode_dwell(sid) >= self.min_dwell.get(m, 1)]
            if not candidates:
                return None
            sid = min(candidates, key=lambda x: x[1])[0]
            return (sid, "OFFLOAD")

        elif action_type == self.ACTION_RECALL_OFFLOAD:
            candidates = [sid for sid, m in modes.items()
                          if m == "OFFLOAD"
                          and sm.get_mode_dwell(sid) >= self.min_dwell.get("OFFLOAD", 1)]
            if not candidates:
                return None
            return (candidates[0], "LITE")

        return None

    def _compute_reward(self, obs):
        reward = 0.0
        for ps in obs["per_stream"].values():
            mode = ps.get("current_mode", "FULL")
            if mode == "SKIP":
                reward += self.r_skip
                continue
            if mode == "OFFLOAD":
                reward += self.r_offload
                continue
            fps = ps.get("infer_fps", ps.get("fps_avg", 0))
            lat = ps.get("infer_latency_p95_ms", ps.get("latency_p95_ms", 0))
            if fps >= self.fps_thresh and lat <= self.lat_thresh:
                reward += self.r_slo_ok
            else:
                if fps < self.fps_thresh:
                    reward += self.r_fps_viol
                if lat > self.lat_thresh:
                    reward += self.r_lat_viol
        return reward
