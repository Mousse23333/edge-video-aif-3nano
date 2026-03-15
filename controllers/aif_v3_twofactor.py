"""
Active Inference (AIF) Controller — v3 Two-Factor Structured Model.

Implements a structured multi-factor Active Inference controller with:
- Factor 1 (S_load):    Local GPU load level (LOW / MEDIUM / HIGH)
- Factor 2 (S_offload): Offload utilization  (NONE / PARTIAL / FULL)

Two observation modalities:
- O_local:  Local inference stream quality (GOOD / MARGINAL / BAD)
            Derived from per-stream fps and latency of FULL/LITE streams.
- O_system: System-wide GPU capacity       (GOOD / MARGINAL / BAD)
            Derived from GPU utilization percentage.

Key design: OFFLOAD actions naturally express cross-factor tradeoffs.
Offloading shifts S_load toward LOW (fewer local inference streams) and
S_offload toward higher utilization, improving O_system predictions
through the LIKELIHOOD_SYSTEM matrix. This resolves the single-factor
model's inability to capture system-level benefits of offloading.

Unlike RL: uses structured generative model + preferences, no training.
Unlike Heuristic: considers uncertainty via beliefs.
Unlike Myopic: considers information gain (epistemic value).
"""

import yaml
import numpy as np
from engine.episode import ControllerInterface


# ── State Spaces ──────────────────────────────────────────────────────────────

# Factor 1: Local GPU load (based on n_infer = local FULL + LITE streams)
LOAD_LOW    = 0   # n_infer <= 3
LOAD_MEDIUM = 1   # n_infer in [4, 5]
LOAD_HIGH   = 2   # n_infer >= 6
N_LOAD_STATES = 3

# Factor 2: Offload utilization
OFFLOAD_NONE    = 0   # 0 streams offloaded
OFFLOAD_PARTIAL = 1   # 1 stream offloaded
OFFLOAD_FULL    = 2   # max streams offloaded (2)
N_OFFLOAD_STATES = 3

# Observation categories (shared by both modalities)
OBS_GOOD     = 0
OBS_MARGINAL = 1
OBS_BAD      = 2
N_OBS_CATS   = 3


# ── Likelihood: P(O_local | S_load, mode) ────────────────────────────────────
# Profiling-derived from Jetson Orin benchmarks (imgsz_results.json).
# O_local categorization:
#   GOOD:     fps >= 13 AND p95 <= 100ms
#   MARGINAL: fps in [10,13) OR p95 in (100,150]
#   BAD:      fps < 10 OR p95 > 150ms
#
# Rows: load states (LOW, MEDIUM, HIGH)
# Cols: obs categories (GOOD, MARGINAL, BAD)

LIKELIHOOD_LOCAL = {
    "FULL": np.array([
        [0.95, 0.04, 0.01],  # LOW:  n=1-3, fps 16-29 → all GOOD
        [0.25, 0.50, 0.25],  # MED:  n=4 borderline, n=5 MARGINAL
        [0.02, 0.13, 0.85],  # HIGH: n=6-8, fps<10 → BAD
    ]),
    "LITE": np.array([
        [0.97, 0.03, 0.00],  # LOW:  n=1-3, fps 21-31 → GOOD
        [0.80, 0.17, 0.03],  # MED:  n=4-5, fps 14-18 → GOOD
        [0.05, 0.55, 0.40],  # HIGH: n=6-7 MARGINAL, n=8 BAD
    ]),
    "SKIP": np.array([
        [0.50, 0.50, 0.00],  # no inference → uninformative
        [0.50, 0.50, 0.00],
        [0.50, 0.50, 0.00],
    ]),
}


# ── Likelihood: P(O_system | S_load, S_offload) ──────────────────────────────
# O_system captures system-wide GPU capacity/health.
# O_system categorization (from GPU utilization):
#   GOOD:     GPU < 60%
#   MARGINAL: GPU 60-80%
#   BAD:      GPU > 80%
#
# Shape: (N_OFFLOAD_STATES, N_LOAD_STATES, N_OBS_CATS)
#
# Key dynamics:
# - HIGH load + no offload   → GPU saturated  → O_system ≈ BAD
# - HIGH load + full offload → GPU relieved   → O_system improves
# - LOW load  + any offload  → GPU fine either way

LIKELIHOOD_SYSTEM = np.array([
    # S_offload = NONE (no streams offloaded)
    [[0.90, 0.08, 0.02],    # S_load=LOW:  GPU comfortable
     [0.15, 0.55, 0.30],    # S_load=MED:  GPU moderate load
     [0.02, 0.13, 0.85]],   # S_load=HIGH: GPU saturated

    # S_offload = PARTIAL (1 stream offloaded)
    [[0.92, 0.06, 0.02],    # LOW+1off:  GPU very comfortable
     [0.45, 0.40, 0.15],    # MED+1off:  GPU improved
     [0.12, 0.48, 0.40]],   # HIGH+1off: partially relieved

    # S_offload = FULL (2 streams offloaded = max)
    [[0.93, 0.05, 0.02],    # LOW+2off:  GPU minimal load
     [0.65, 0.28, 0.07],    # MED+2off:  GPU comfortable
     [0.30, 0.48, 0.22]],   # HIGH+2off: significantly relieved
])


# ── Transition Models ─────────────────────────────────────────────────────────

def transition_load(s, a_type):
    """
    P(S_load' | S_load=s, action_type).
    a_type: 'demote' (FULL->LITE, *->SKIP, *->OFFLOAD) decreases load
            'promote' (SKIP->LITE, LITE->FULL, OFFLOAD->LITE) increases load
            'noop': mostly stays same
    """
    p = np.zeros(N_LOAD_STATES)
    if a_type == 'noop':
        p[s] = 0.9
        if s > 0: p[s-1] = 0.05
        if s < N_LOAD_STATES-1: p[s+1] = 0.05
        if s == 0: p[0] += 0.05
        if s == N_LOAD_STATES-1: p[-1] += 0.05
    elif a_type == 'demote':
        if s > 0:
            p[s-1] = 0.75
            p[s]   = 0.25
        else:
            p[0] = 1.0
    elif a_type == 'promote':
        if s < N_LOAD_STATES - 1:
            p[s+1] = 0.75
            p[s]   = 0.25
        else:
            p[-1] = 1.0
    return p


def transition_offload(s_off, a_offload):
    """
    P(S_offload' | S_offload=s_off, action).
    a_offload: 'add' (stream → OFFLOAD), 'remove' (OFFLOAD → local),
               'none' (no offload change)
    """
    p = np.zeros(N_OFFLOAD_STATES)
    if a_offload == 'add':
        new_s = min(s_off + 1, N_OFFLOAD_STATES - 1)
        p[new_s] = 0.90
        p[s_off] = 0.10
    elif a_offload == 'remove':
        new_s = max(s_off - 1, 0)
        p[new_s] = 0.90
        p[s_off] = 0.10
    else:
        p[s_off] = 1.0
    return p


# ── AIF Controller ────────────────────────────────────────────────────────────

class AIFController(ControllerInterface):
    """
    Two-factor Active Inference controller.

    Minimizes Expected Free Energy (EFE) considering both:
    - O_local:  local stream quality   (pragmatic_local)
    - O_system: system GPU capacity    (pragmatic_system)

    EFE(a) = pragmatic_local + w_sys * pragmatic_system
             - w_epi * epistemic + switch_penalty

    OFFLOAD naturally emerges as beneficial at high load because the
    system-level factor captures the GPU relief effect through
    LIKELIHOOD_SYSTEM, without needing a hand-tuned OFFLOAD likelihood.
    """

    def __init__(self, config_dir="/app/config"):
        with open(f"{config_dir}/slo.yaml") as f:
            slo = yaml.safe_load(f)
        with open(f"{config_dir}/switch_cost.yaml") as f:
            self.switch_cfg = yaml.safe_load(f)
        with open(f"{config_dir}/action_space.yaml") as f:
            action_cfg = yaml.safe_load(f)

        hc = slo["hard_constraints"]
        self.lat_thresh  = hc["p95_latency_ms"]["threshold"]
        self.fps_thresh  = hc["min_fps"]["threshold"]
        self.max_skip    = hc["max_skip_ratio"]["threshold"]
        self.min_dwell   = self.switch_cfg.get("min_dwell_before_exit", {})

        # OFFLOAD config
        offload_cfg = action_cfg.get("modes", {}).get("OFFLOAD", {})
        self.offload_enabled = offload_cfg.get("enabled", False)
        offload_urls = offload_cfg.get("urls", [])
        self.max_offload = len(offload_urls) if self.offload_enabled else 0

        # Preferred observations (two modalities)
        self.C_local  = np.array([0.85, 0.12, 0.03])  # prefer GOOD local quality
        self.C_system = np.array([0.80, 0.15, 0.05])  # prefer low GPU utilization

        # Two-factor belief (mean-field factorization)
        self.belief_load    = np.ones(N_LOAD_STATES) / N_LOAD_STATES
        self.belief_offload = np.ones(N_OFFLOAD_STATES) / N_OFFLOAD_STATES

        self.modes = ["FULL", "LITE", "SKIP"]
        if self.offload_enabled:
            self.modes.append("OFFLOAD")

        # Hyperparameters
        self.precision       = 4.0   # softmax sharpness for EFE
        self.system_weight   = 0.5   # weight for system-level pragmatic
        self.epistemic_weight = 0.3  # weight for information gain

        # ASAP temporal smoothing
        self.asap_weight = 0.6

        # Anti-oscillation tracking
        self._last_switch = {}
        self._step = 0
        self._global_last_switch_step = -3
        self._prev_action_key = "noop"

    def on_episode_start(self, config):
        self.belief_load    = np.ones(N_LOAD_STATES) / N_LOAD_STATES
        self.belief_offload = np.ones(N_OFFLOAD_STATES) / N_OFFLOAD_STATES
        self._last_switch = {}
        self._step = 0
        self._global_last_switch_step = -3
        self._prev_action_key = "noop"

    def select_action(self, obs, stream_manager):
        per_stream = obs["per_stream"]
        n_active = obs["global"]["n_active_streams"]
        if n_active == 0:
            return None

        # 1. Update both beliefs from current observation
        self._update_beliefs(obs, per_stream)

        # 2. Get valid actions
        stream_ids = sorted(per_stream.keys())
        actions = self._enumerate_actions(per_stream, stream_ids,
                                          stream_manager, n_active)

        # 3. Compute EFE for each action
        efe_scores = {}
        for action_key, action in actions.items():
            efe = self._compute_efe(action_key, obs, per_stream, actions)
            efe_scores[action_key] = efe

        # 4. Precision-weighted softmax with anti-oscillation
        keys   = list(efe_scores.keys())
        scores = np.array([efe_scores[k] for k in keys])

        # ASAP temporal smoothing: penalize reversing previous action
        for i, k in enumerate(keys):
            if k != "noop" and k != self._prev_action_key:
                action_val = actions.get(k)
                if action_val and self._prev_action_key != "noop":
                    prev_val = actions.get(self._prev_action_key)
                    if prev_val:
                        prev_sid, prev_new = prev_val
                        cur_sid, cur_new = action_val
                        if prev_sid == cur_sid:
                            prev_old = per_stream.get(prev_sid, {}).get(
                                "current_mode", "FULL")
                            if cur_new == prev_old:
                                scores[i] += self.asap_weight

        # Global cooldown: hard block for 3 steps after a switch
        steps_since_switch = self._step - self._global_last_switch_step
        if steps_since_switch < 3:
            best_key = "noop"
        else:
            neg_efe = -self.precision * (scores - scores.min())
            probs   = np.exp(neg_efe)
            probs  /= probs.sum()
            best_key = np.random.choice(keys, p=probs)

        result = actions[best_key]
        self._prev_action_key = best_key

        if result is not None:
            sid, new_mode = result
            old_mode = per_stream.get(sid, {}).get("current_mode", "FULL")
            self._last_switch[sid] = (old_mode, new_mode, self._step)
            self._global_last_switch_step = self._step
        self._step += 1

        return result

    # ── Belief Update ─────────────────────────────────────────────────────────

    def _update_beliefs(self, obs, per_stream):
        """Update both belief factors from current observations."""

        # ── Factor 1: S_load from O_local ──
        obs_cats = []
        for sid, ps in per_stream.items():
            mode = ps.get("current_mode", "FULL")
            if mode in ("SKIP", "OFFLOAD"):
                continue
            fps = ps.get("infer_fps", ps.get("fps_avg", 0))
            lat = ps.get("infer_latency_p95_ms", ps.get("latency_p95_ms", 0))
            obs_cats.append(self._categorize_obs(fps, lat))

        if obs_cats:
            obs_local = int(np.round(np.mean(obs_cats)))
            rep_mode = self._get_dominant_local_mode(per_stream)
            likelihood_vec = LIKELIHOOD_LOCAL[rep_mode][:, obs_local]
            posterior = self.belief_load * likelihood_vec
            total = posterior.sum()
            if total > 1e-10:
                self.belief_load = posterior / total
            else:
                self.belief_load = np.ones(N_LOAD_STATES) / N_LOAD_STATES

        # Cross-validate S_load from O_system (GPU utilization)
        gpu_util = obs["global"].get("gpu_util_avg", 0)
        obs_system = self._categorize_gpu(gpu_util)
        # P(O_system | S_load) marginalized over current S_offload belief
        sys_likelihood = np.zeros(N_LOAD_STATES)
        for s_off in range(N_OFFLOAD_STATES):
            sys_likelihood += self.belief_offload[s_off] * \
                              LIKELIHOOD_SYSTEM[s_off, :, obs_system]
        posterior2 = self.belief_load * sys_likelihood
        total2 = posterior2.sum()
        if total2 > 1e-10:
            self.belief_load = posterior2 / total2

        # ── Factor 2: S_offload from observed offload count ──
        # Directly observable, so set near-deterministically
        n_offload = sum(1 for ps in per_stream.values()
                        if ps.get("current_mode") == "OFFLOAD")
        if n_offload == 0:
            s_off = OFFLOAD_NONE
        elif n_offload < self.max_offload:
            s_off = OFFLOAD_PARTIAL
        else:
            s_off = OFFLOAD_FULL
        self.belief_offload = np.full(N_OFFLOAD_STATES, 0.03)
        self.belief_offload[s_off] = 0.94
        self.belief_offload /= self.belief_offload.sum()

    # ── EFE Computation ───────────────────────────────────────────────────────

    def _compute_efe(self, action_key, obs, per_stream, actions=None):
        """
        Two-factor EFE.

        EFE(a) = pragmatic_local + w_sys * pragmatic_system
                 - w_epi * epistemic + switch_penalty

        For OFFLOAD:
        - S_load transitions toward LOW (demote)
        - S_offload transitions toward PARTIAL/FULL (add)
        - O_local predicted from remaining local streams (improves)
        - O_system predicted from (S_load', S_offload') via LIKELIHOOD_SYSTEM
          → captures GPU relief benefit naturally
        """
        # Determine action details
        if action_key == "noop":
            a_type_load = "noop"
            a_type_offload = "none"
            sid, new_mode = None, None
        else:
            action_val = actions.get(action_key) if actions else None
            if action_val:
                sid, new_mode = action_val
            else:
                parts = action_key.split("_")
                sid = int(parts[0])
                new_mode = parts[1]
            old_mode = per_stream.get(sid, {}).get("current_mode", "FULL")
            a_type_load = self._action_type(old_mode, new_mode)

            if new_mode == "OFFLOAD":
                a_type_offload = "add"
            elif old_mode == "OFFLOAD":
                a_type_offload = "remove"
            else:
                a_type_offload = "none"

        # ── Predicted state distributions after action ──

        # Factor 1: S_load'
        Q_load = np.zeros(N_LOAD_STATES)
        for s in range(N_LOAD_STATES):
            Q_load += self.belief_load[s] * transition_load(s, a_type_load)

        # Factor 2: S_offload'
        Q_offload = np.zeros(N_OFFLOAD_STATES)
        for s in range(N_OFFLOAD_STATES):
            Q_offload += self.belief_offload[s] * \
                         transition_offload(s, a_type_offload)

        # ── O_local: predict local stream observation ──
        # For OFFLOAD/SKIP: use remaining local streams' dominant mode
        # For FULL/LITE: use the new mode
        if new_mode in ("OFFLOAD", "SKIP"):
            sim_mode = self._get_dominant_local_mode(per_stream,
                                                     exclude_sid=sid)
        elif new_mode in ("FULL", "LITE"):
            sim_mode = new_mode
        else:  # noop
            sim_mode = self._get_dominant_local_mode(per_stream)

        likelihood_local = LIKELIHOOD_LOCAL[sim_mode]
        P_o_local = Q_load @ likelihood_local  # (N_OBS_CATS,)
        P_o_local = np.clip(P_o_local, 1e-10, 1.0)

        # ── O_system: predict system GPU capacity ──
        # Marginalize over both predicted factors:
        # P(O_sys|a) = Σ_{s_load,s_off} Q(s_load|a) Q(s_off|a) P(O_sys|s_load,s_off)
        P_o_system = np.zeros(N_OBS_CATS)
        for s_load in range(N_LOAD_STATES):
            for s_off in range(N_OFFLOAD_STATES):
                P_o_system += Q_load[s_load] * Q_offload[s_off] * \
                              LIKELIHOOD_SYSTEM[s_off, s_load, :]
        P_o_system = np.clip(P_o_system, 1e-10, 1.0)

        # ── Pragmatic values ──
        C_local_safe  = np.clip(self.C_local,  1e-10, 1.0)
        C_system_safe = np.clip(self.C_system, 1e-10, 1.0)

        pragmatic_local  = -np.sum(P_o_local  * np.log(C_local_safe))
        pragmatic_system = -np.sum(P_o_system * np.log(C_system_safe))

        # ── Epistemic value (information gain about S_load from O_local) ──
        epistemic = 0.0
        for o in range(N_OBS_CATS):
            if P_o_local[o] < 1e-10:
                continue
            Q_s_given_o = Q_load * likelihood_local[:, o]
            norm = Q_s_given_o.sum()
            if norm > 1e-10:
                Q_s_given_o /= norm
            else:
                continue
            ratio = np.clip(Q_s_given_o / np.clip(Q_load, 1e-10, 1),
                            1e-10, 1e10)
            kl = np.sum(Q_s_given_o * np.log(ratio))
            epistemic += P_o_local[o] * kl

        # ── Switch cost penalty ──
        switch_penalty = 0.0
        if action_key != "noop" and new_mode is not None:
            old_mode = per_stream.get(sid, {}).get("current_mode", "FULL")
            key = f"{old_mode}->{new_mode}"
            t = self.switch_cfg.get("transitions", {}).get(key, {})
            switch_penalty = t.get("penalty_score", 0) * 0.05

            # Anti-oscillation: heavy penalty for reversing a recent switch
            last = self._last_switch.get(sid)
            if last is not None:
                last_from, last_to, last_step = last
                steps_ago = self._step - last_step
                if new_mode == last_from and steps_ago <= 5:
                    switch_penalty += 0.8 * (1.0 - steps_ago / 6.0)

        efe = pragmatic_local \
              + self.system_weight * pragmatic_system \
              - self.epistemic_weight * epistemic \
              + switch_penalty
        return efe

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _categorize_obs(self, fps, lat):
        """Categorize local stream observation."""
        if fps >= 13 and lat <= 100:
            return OBS_GOOD
        elif fps >= self.fps_thresh and lat <= self.lat_thresh:
            return OBS_MARGINAL
        return OBS_BAD

    def _categorize_gpu(self, gpu_util):
        """Categorize system GPU observation."""
        if gpu_util < 60:
            return OBS_GOOD
        elif gpu_util < 80:
            return OBS_MARGINAL
        return OBS_BAD

    def _get_dominant_local_mode(self, per_stream, exclude_sid=None):
        """Get the dominant mode among local inference streams."""
        n_full = sum(1 for sid, ps in per_stream.items()
                     if sid != exclude_sid
                     and ps.get("current_mode") == "FULL")
        n_lite = sum(1 for sid, ps in per_stream.items()
                     if sid != exclude_sid
                     and ps.get("current_mode") == "LITE")
        if n_full >= n_lite and n_full > 0:
            return "FULL"
        elif n_lite > 0:
            return "LITE"
        return "FULL"

    def _enumerate_actions(self, per_stream, stream_ids, sm, n_active):
        """Returns dict of action_key -> (stream_id, mode) or None (NO-OP)."""
        actions = {"noop": None}
        n_skip = sum(1 for ps in per_stream.values()
                     if ps.get("current_mode") == "SKIP")
        n_offload = sum(1 for ps in per_stream.values()
                        if ps.get("current_mode") == "OFFLOAD")

        for sid in stream_ids:
            old_mode = per_stream[sid].get("current_mode", "FULL")
            dwell = sm.get_mode_dwell(sid)
            min_d = self.min_dwell.get(old_mode, 1)
            if dwell < min_d:
                continue
            for new_mode in self.modes:
                if new_mode == old_mode:
                    continue
                if new_mode == "SKIP":
                    if (n_skip + 1) / n_active > self.max_skip:
                        continue
                if new_mode == "OFFLOAD":
                    if n_offload >= self.max_offload:
                        continue
                key = f"{sid}_{new_mode}"
                actions[key] = (sid, new_mode)
        return actions

    def _action_type(self, old_mode, new_mode):
        """Classify action effect on local GPU load."""
        rank = {"FULL": 2, "LITE": 1, "SKIP": 0, "OFFLOAD": 0}
        if rank.get(new_mode, 1) < rank.get(old_mode, 1):
            return "demote"
        elif rank.get(new_mode, 1) > rank.get(old_mode, 1):
            return "promote"
        return "noop"
