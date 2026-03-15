"""
REINFORCE (Policy Gradient) Controller.

Learns a stochastic policy π(a|s) directly using REINFORCE with baseline.
More sample-efficient than DQN for short episodes because it uses full
episode returns (Monte Carlo), not TD bootstrapping.

Architecture: linear softmax policy over discretized state features.
Baseline: running average reward (reduces variance).

Key difference from Tabular QL: policy gradient (continuous update direction)
vs value-based (max over Q-table). Handles state generalization better.
Key difference from AIF: learns from scalar reward signal (no generative model).
"""

import yaml
import numpy as np
import random
from engine.episode import ControllerInterface

# Shared state discretization with AIF/Tabular
MODES = ["FULL", "LITE", "SKIP"]

# Abstract action types (same as Tabular QL for fair comparison)
ACTION_NOOP           = 0
ACTION_DEMOTE_FULL    = 1
ACTION_DEMOTE_LITE    = 2
ACTION_PROMOTE_SKIP   = 3
ACTION_PROMOTE_LITE   = 4
ACTION_OFFLOAD        = 5
ACTION_RECALL_OFFLOAD = 6
N_ACTIONS = 7


def extract_features(obs):
    """
    Linear feature vector for policy.
    Returns numpy array of shape (feature_dim,).
    """
    per_stream = obs["per_stream"]
    gl = obs["global"]

    fps_list = [ps.get("infer_fps", ps.get("fps_avg", 0))
                for ps in per_stream.values()
                if ps.get("current_mode") not in ("SKIP",)]
    lat_list = [ps.get("infer_latency_p95_ms", ps.get("latency_p95_ms", 0))
                for ps in per_stream.values()
                if ps.get("current_mode") not in ("SKIP",)]

    avg_fps = np.mean(fps_list) if fps_list else 0
    avg_lat = np.mean(lat_list) if lat_list else 0
    n_full  = sum(1 for ps in per_stream.values() if ps.get("current_mode") == "FULL")
    n_lite  = sum(1 for ps in per_stream.values() if ps.get("current_mode") == "LITE")
    n_skip  = sum(1 for ps in per_stream.values() if ps.get("current_mode") == "SKIP")
    n_total = gl.get("n_active_streams", 1)
    gpu     = gl.get("gpu_util_avg", 0)

    n_offload = sum(1 for ps in per_stream.values() if ps.get("current_mode") == "OFFLOAD")

    feat = np.array([
        avg_fps / 30.0,                      # normalized fps
        min(avg_lat, 300) / 300.0,           # normalized latency
        n_full / max(n_total, 1),            # fraction FULL
        n_lite / max(n_total, 1),            # fraction LITE
        n_skip / max(n_total, 1),            # fraction SKIP
        n_offload / max(n_total, 1),         # fraction OFFLOAD
        gpu / 100.0,                          # gpu util
        n_total / 8.0,                        # normalized stream count
        float(avg_fps < 10),                  # SLO FPS violation indicator
        float(avg_lat > 150),                 # SLO lat violation indicator
        1.0,                                  # bias
    ], dtype=np.float32)
    return feat


class REINFORCEController(ControllerInterface):
    """
    Online REINFORCE with linear softmax policy and running-average baseline.
    Collects (state, action, reward) trajectory per episode, updates at end.
    Between episodes, carries over learned weights (warm-starting).
    """

    FEATURE_DIM = 11  # added offload fraction

    def __init__(self, config_dir="/app/config",
                 lr=0.05, gamma=0.95, entropy_coeff=0.01):
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

        # OFFLOAD config
        with open(f"{config_dir}/action_space.yaml") as f:
            action_cfg = yaml.safe_load(f)
        offload_cfg = action_cfg.get("modes", {}).get("OFFLOAD", {})
        self.offload_enabled = offload_cfg.get("enabled", False)
        offload_urls = offload_cfg.get("urls", [])
        self.max_offload = len(offload_urls) if self.offload_enabled else 0

        self.lr            = lr
        self.gamma         = gamma
        self.entropy_coeff = entropy_coeff
        self._train_entropy_coeff = entropy_coeff
        self._eval_entropy_coeff  = 0.001

        # Linear policy weights: shape (FEATURE_DIM, N_ACTIONS)
        # PERSISTS across episodes for cross-episode learning
        self.theta = np.zeros((self.FEATURE_DIM, N_ACTIONS), dtype=np.float32)

        # Running baseline (average reward) — persists for stability
        self.baseline = 0.0
        self.baseline_alpha = 0.1

        # Episode trajectory buffer
        self._trajectory = []   # list of (features, action_type, valid_mask)
        self._rewards    = []
        self._eval_mode  = False
        self._episode    = 0

    def set_eval_mode(self, eval_mode=True):
        """Switch between training (more entropy) and evaluation (near-greedy)."""
        self._eval_mode = eval_mode
        self.entropy_coeff = self._eval_entropy_coeff if eval_mode else self._train_entropy_coeff

    def on_episode_start(self, config):
        # Reset per-episode trajectory only; theta and baseline persist
        self._trajectory = []
        self._rewards    = []
        self._episode += 1

    def on_episode_end(self, history):
        """Update policy using REINFORCE at episode end."""
        if len(self._rewards) < 2:
            return

        # Compute discounted returns G_t
        T = len(self._rewards)
        G = np.zeros(T)
        G[-1] = self._rewards[-1]
        for t in range(T - 2, -1, -1):
            G[t] = self._rewards[t] + self.gamma * G[t + 1]

        # Update baseline
        avg_G = float(np.mean(G))
        self.baseline = (1 - self.baseline_alpha) * self.baseline + \
                        self.baseline_alpha * avg_G

        # Policy gradient update
        for t, (feat, action_type, valid_mask) in enumerate(self._trajectory):
            advantage = G[t] - self.baseline

            # Compute policy logits and probabilities
            logits = feat @ self.theta                    # (N_ACTIONS,)
            logits[~valid_mask] = -1e9
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            probs = np.clip(probs, 1e-8, 1.0)

            # REINFORCE gradient: ∇log π(a|s) * advantage
            one_hot = np.zeros(N_ACTIONS)
            one_hot[action_type] = 1.0
            grad_log_pi = one_hot - probs            # (N_ACTIONS,)

            # Entropy gradient (encourages exploration)
            grad_entropy = -(np.log(probs) + 1.0)   # (N_ACTIONS,)

            # Update: θ += lr * (advantage * ∇logπ + entropy_coeff * ∇H)
            update = advantage * grad_log_pi + self.entropy_coeff * grad_entropy
            self.theta += self.lr * np.outer(feat, update)

    def select_action(self, obs, stream_manager):
        per_stream = obs["per_stream"]
        n_active   = obs["global"]["n_active_streams"]
        if n_active == 0:
            return None

        feat       = extract_features(obs)
        valid_mask = self._get_valid_mask(per_stream, stream_manager, n_active)

        # Compute stochastic policy
        logits = feat @ self.theta
        logits[~valid_mask] = -1e9
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= probs.sum()

        action_type = int(np.random.choice(N_ACTIONS, p=probs))

        # Record transition
        reward = self._compute_reward(obs)
        self._trajectory.append((feat, action_type, valid_mask.copy()))
        self._rewards.append(reward)

        return self._resolve_action(action_type, per_stream, stream_manager, n_active)

    def _get_valid_mask(self, per_stream, sm, n_active):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[ACTION_NOOP] = True
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
                mask[ACTION_DEMOTE_FULL] = True
                if self.offload_enabled and n_offload < self.max_offload:
                    mask[ACTION_OFFLOAD] = True
            elif mode == "LITE":
                if (n_skip + 1) / n_active <= self.max_skip:
                    mask[ACTION_DEMOTE_LITE] = True
                mask[ACTION_PROMOTE_LITE] = True
                if self.offload_enabled and n_offload < self.max_offload:
                    mask[ACTION_OFFLOAD] = True
            elif mode == "SKIP":
                mask[ACTION_PROMOTE_SKIP] = True
            elif mode == "OFFLOAD":
                mask[ACTION_RECALL_OFFLOAD] = True
        return mask

    def _resolve_action(self, action_type, per_stream, sm, n_active):
        if action_type == ACTION_NOOP:
            return None

        modes = {sid: ps.get("current_mode", "FULL")
                 for sid, ps in per_stream.items()}
        fps   = {sid: ps.get("infer_fps", ps.get("fps_avg", 0))
                 for sid, ps in per_stream.items()}
        n_skip = sum(1 for m in modes.values() if m == "SKIP")

        if action_type == ACTION_DEMOTE_FULL:
            cands = [(sid, fps[sid]) for sid, m in modes.items()
                     if m == "FULL" and sm.get_mode_dwell(sid) >= self.min_dwell.get("FULL", 1)]
            if not cands:
                return None
            return (min(cands, key=lambda x: x[1])[0], "LITE")

        elif action_type == ACTION_DEMOTE_LITE:
            if (n_skip + 1) / n_active > self.max_skip:
                return None
            cands = [(sid, fps[sid]) for sid, m in modes.items()
                     if m == "LITE" and sm.get_mode_dwell(sid) >= self.min_dwell.get("LITE", 1)]
            if not cands:
                return None
            return (min(cands, key=lambda x: x[1])[0], "SKIP")

        elif action_type == ACTION_PROMOTE_SKIP:
            cands = [sid for sid, m in modes.items()
                     if m == "SKIP" and sm.get_mode_dwell(sid) >= self.min_dwell.get("SKIP", 2)]
            if not cands:
                return None
            return (cands[0], "LITE")

        elif action_type == ACTION_PROMOTE_LITE:
            cands = [sid for sid, m in modes.items()
                     if m == "LITE" and sm.get_mode_dwell(sid) >= self.min_dwell.get("LITE", 1)]
            if not cands:
                return None
            return (cands[0], "FULL")

        elif action_type == ACTION_OFFLOAD:
            cands = [(sid, fps[sid]) for sid, m in modes.items()
                     if m in ("FULL", "LITE")
                     and sm.get_mode_dwell(sid) >= self.min_dwell.get(m, 1)]
            if not cands:
                return None
            return (min(cands, key=lambda x: x[1])[0], "OFFLOAD")

        elif action_type == ACTION_RECALL_OFFLOAD:
            cands = [sid for sid, m in modes.items()
                     if m == "OFFLOAD"
                     and sm.get_mode_dwell(sid) >= self.min_dwell.get("OFFLOAD", 1)]
            if not cands:
                return None
            return (cands[0], "LITE")

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
