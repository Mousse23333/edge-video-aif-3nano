"""
DQN (Deep Q-Network) Controller.

Online RL baseline. Learns a Q-function mapping (state, action) -> expected reward.
Shares the exact same observation/action interface as AIF and Heuristic.

Architecture: small MLP (3 layers), epsilon-greedy exploration,
experience replay buffer, target network.

Key difference from AIF: maximizes scalar reward (requires reward engineering),
reactive learner, no explicit belief or generative model.
"""

import yaml
import random
import numpy as np
from collections import deque
from engine.episode import ControllerInterface

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Q-Network ────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── DQN Controller ────────────────────────────────────────────────────────────

class DQNController(ControllerInterface):
    """
    Online DQN. Learns during the episode (no pre-training).
    Uses same reward structure as slo.yaml to stay comparable.
    """

    def __init__(self, config_dir="/app/config",
                 lr=1e-3, gamma=0.95, epsilon_start=0.15,
                 epsilon_end=0.02, epsilon_decay=20,
                 replay_size=500, batch_size=32,
                 target_update_freq=10, hidden=64):

        with open(f"{config_dir}/slo.yaml") as f:
            slo = yaml.safe_load(f)
        with open(f"{config_dir}/switch_cost.yaml") as f:
            self.switch_cfg = yaml.safe_load(f)

        hc = slo["hard_constraints"]
        self.lat_thresh = hc["p95_latency_ms"]["threshold"]
        self.fps_thresh = hc["min_fps"]["threshold"]
        self.max_skip_ratio = hc["max_skip_ratio"]["threshold"]
        self.min_dwell = self.switch_cfg.get("min_dwell_before_exit", {})

        rl_r = slo["rl_reward"]
        self.r_slo_ok    = rl_r["slo_satisfied_bonus"]
        self.r_lat_viol  = rl_r["latency_violation_penalty"]
        self.r_fps_viol  = rl_r["fps_violation_penalty"]
        self.r_skip      = rl_r["skip_penalty"]
        self.r_offload   = rl_r.get("offload_penalty", -0.2)
        self.r_sw_mult   = rl_r["switch_penalty_multiplier"]
        self.r_noop      = rl_r["no_op_bonus"]

        # OFFLOAD config
        with open(f"{config_dir}/action_space.yaml") as f:
            action_cfg = yaml.safe_load(f)
        offload_cfg = action_cfg.get("modes", {}).get("OFFLOAD", {})
        self.offload_enabled = offload_cfg.get("enabled", False)
        offload_urls = offload_cfg.get("urls", [])
        self.max_offload = len(offload_urls) if self.offload_enabled else 0

        self.modes = ["FULL", "LITE", "SKIP"]
        if self.offload_enabled:
            self.modes.append("OFFLOAD")
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.replay = deque(maxlen=replay_size)

        # State/action dims depend on max streams (set at episode start)
        self.max_streams = 8
        self.n_modes = len(self.modes)

        # state_dim = per-stream: (fps, p95, mode_onehot×n_modes) + global: (gpu, n_active)
        self.state_dim = self.max_streams * (2 + self.n_modes) + 2
        # action_dim = N_streams × n_modes + 1 (NO-OP)
        self.action_dim = self.max_streams * self.n_modes + 1

        self._step = 0
        self._total_steps = 0
        self._prev_state = None
        self._prev_action_idx = None
        self._eval_mode = False

        if HAS_TORCH:
            # Network, replay buffer, optimizer PERSIST across episodes
            self.q_net = QNetwork(self.state_dim, self.action_dim, hidden)
            self.target_net = QNetwork(self.state_dim, self.action_dim, hidden)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss()

    def set_eval_mode(self, eval_mode=True):
        """Switch between training (exploration) and evaluation (exploitation)."""
        self._eval_mode = eval_mode

    def on_episode_start(self, config):
        # Reset per-episode state only; network and replay buffer persist
        self._step = 0
        self._prev_state = None
        self._prev_action_idx = None

    def select_action(self, obs, stream_manager):
        if not HAS_TORCH:
            return None

        per_stream = obs["per_stream"]
        n_active = obs["global"]["n_active_streams"]
        if n_active == 0:
            return None

        state_vec = self._encode_state(obs)

        # Compute reward for previous transition
        if self._prev_state is not None and self._prev_action_idx is not None:
            reward = self._compute_reward(obs, stream_manager)
            done = False
            self.replay.append((
                self._prev_state, self._prev_action_idx,
                reward, state_vec, done
            ))
            self._learn()

        # Epsilon: eval mode uses near-greedy, training decays across total steps
        if self._eval_mode:
            eps = self.epsilon_end
        else:
            eps = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
                  np.exp(-self._total_steps / (self.epsilon_decay * 5))

        stream_ids = sorted(per_stream.keys())
        valid_actions = self._get_valid_actions(per_stream, stream_ids, stream_manager, n_active)

        if random.random() < eps or not valid_actions:
            action_idx = random.choice(valid_actions + [self.action_dim - 1])
        else:
            with torch.no_grad():
                q_vals = self.q_net(torch.FloatTensor(state_vec).unsqueeze(0))[0]
            # Mask invalid actions
            mask = torch.full((self.action_dim,), float('-inf'))
            for idx in valid_actions + [self.action_dim - 1]:
                mask[idx] = 0
            q_masked = q_vals + mask
            action_idx = int(q_masked.argmax().item())

        self._prev_state = state_vec
        self._prev_action_idx = action_idx
        self._step += 1
        self._total_steps += 1

        if self._step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return self._decode_action(action_idx, sorted(per_stream.keys()))

    def _encode_state(self, obs):
        per_feat = 2 + self.n_modes  # fps, lat, mode_onehot
        vec = np.zeros(self.state_dim, dtype=np.float32)
        stream_ids = sorted(obs["per_stream"].keys())
        for i, sid in enumerate(stream_ids[:self.max_streams]):
            ps = obs["per_stream"][sid]
            base = i * per_feat
            vec[base + 0] = ps.get("infer_fps", ps.get("fps_avg", 0)) / 30.0
            vec[base + 1] = min(ps.get("infer_latency_p95_ms", ps.get("latency_p95_ms", 0)), 300) / 300.0
            mode = ps.get("current_mode", "FULL")
            mode_idx = self.modes.index(mode) if mode in self.modes else 0
            vec[base + 2 + mode_idx] = 1.0
        global_base = self.max_streams * per_feat
        vec[global_base + 0] = obs["global"].get("gpu_util_avg", 0) / 100.0
        vec[global_base + 1] = obs["global"].get("n_active_streams", 0) / self.max_streams
        return vec

    def _get_valid_actions(self, per_stream, stream_ids, sm, n_active):
        valid = []
        n_offload = sum(1 for ps in per_stream.values()
                        if ps.get("current_mode") == "OFFLOAD")
        for i, sid in enumerate(stream_ids[:self.max_streams]):
            old_mode = per_stream[sid].get("current_mode", "FULL")
            dwell = sm.get_mode_dwell(sid)
            min_d = self.min_dwell.get(old_mode, 1)
            if dwell < min_d:
                continue
            for j, new_mode in enumerate(self.modes):
                if new_mode == old_mode:
                    continue
                if new_mode == "SKIP":
                    n_skip = sum(1 for ps in per_stream.values()
                                 if ps.get("current_mode") == "SKIP")
                    if (n_skip + 1) / n_active > self.max_skip_ratio:
                        continue
                if new_mode == "OFFLOAD":
                    if n_offload >= self.max_offload:
                        continue
                action_idx = i * self.n_modes + j
                valid.append(action_idx)
        return valid

    def _decode_action(self, action_idx, stream_ids):
        if action_idx == self.action_dim - 1:
            return None  # NO-OP
        stream_pos = action_idx // self.n_modes
        mode_idx = action_idx % self.n_modes
        if stream_pos >= len(stream_ids):
            return None
        return (stream_ids[stream_pos], self.modes[mode_idx])

    def _compute_reward(self, obs, sm):
        reward = 0.0
        for sid, ps in obs["per_stream"].items():
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
                if lat > self.lat_thresh:
                    reward += self.r_lat_viol
                if fps < self.fps_thresh:
                    reward += self.r_fps_viol
        return reward

    def _learn(self):
        if len(self.replay) < self.batch_size:
            return
        batch = random.sample(self.replay, self.batch_size)
        s, a, r, s2, d = zip(*batch)
        s  = torch.FloatTensor(np.array(s))
        a  = torch.LongTensor(np.array(a))
        r  = torch.FloatTensor(np.array(r))
        s2 = torch.FloatTensor(np.array(s2))
        d  = torch.FloatTensor(np.array(d))

        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(s2).max(1)[0]
        q_target = r + self.gamma * q_next * (1 - d)

        loss = self.loss_fn(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
