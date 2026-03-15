"""
Active Inference (AIF) Controller — v1 (System-Level OFFLOAD Likelihood).

Implements a discrete-state POMDP using Active Inference principles:
- Maintains a belief distribution over hidden system load state
- Selects actions by minimizing Expected Free Energy (EFE)
- EFE = Epistemic value (information gain) + Pragmatic value (goal-seeking)

Hidden state: system load level (LOW / MEDIUM / HIGH), based on n_infer.
Observation: per-stream (fps, latency) categorized as GOOD/MARGINAL/BAD.

Key design: OFFLOAD likelihood reflects *system-level* effect (GPU relief
for remaining streams), NOT the offloaded stream's own poor metrics.
This is validated by v0→v1 sensitivity analysis: correcting OFFLOAD
semantics from per-stream BAD to system-level improves SLO from 78%→91%.

Unlike Heuristic: considers uncertainty via beliefs, not just thresholds.
Unlike Myopic Greedy: considers information gain (epistemic value).
Unlike RL: uses preferences (C) instead of reward function, no training.
"""

import yaml
import numpy as np
from engine.episode import ControllerInterface


# ── State / Observation Spaces ────────────────────────────────────────────────

# Hidden state: system load level
LOAD_LOW    = 0   # n_infer <= 3
LOAD_MEDIUM = 1   # n_infer in [4, 5]
LOAD_HIGH   = 2   # n_infer >= 6
N_LOAD_STATES = 3

# Observation categories per stream
OBS_GOOD    = 0   # fps >= 13 AND lat <= 100ms
OBS_MARGINAL = 1  # fps in [10,13) OR lat in (100,150]
OBS_BAD     = 2   # fps < 10 OR lat > 150ms
N_OBS_CATS  = 3

# ── Profiling-derived likelihood P(obs | load_state, mode) ────────────────────
# Fitted from empirical benchmarks on Jetson Orin (imgsz_results.json):
#
#   FULL (imgsz=640) profiling:
#     n=1: fps=28.9, p95=34ms   n=2: fps=21.6, p95=48ms   n=3: fps=16.7, p95=59ms
#     n=4: fps=13.0, p95=75ms   n=5: fps=10.6, p95=92ms
#     n=6: fps=8.7,  p95=107ms  n=7: fps=7.9,  p95=113ms  n=8: fps=7.2, p95=130ms
#
#   LITE (imgsz=320) profiling:
#     n=1: fps=30.8, p95=31ms   n=2: fps=25.5, p95=36ms   n=3: fps=21.2, p95=41ms
#     n=4: fps=17.6, p95=47ms   n=5: fps=14.7, p95=57ms
#     n=6: fps=12.4, p95=66ms   n=7: fps=10.8, p95=74ms   n=8: fps=9.4, p95=87ms
#
# Observation categorization:
#   GOOD:     fps >= 13 AND p95 <= 100ms
#   MARGINAL: fps in [10,13) OR p95 in (100,150]
#   BAD:      fps < 10 OR p95 > 150ms
#
# Load states: LOW = n_infer in {1,2,3}, MEDIUM = {4,5}, HIGH = {6,7,8+}
#
# Rows: load states (LOW, MEDIUM, HIGH)
# Cols: obs categories (GOOD, MARGINAL, BAD)
# v0 OFFLOAD likelihood (per-stream BAD — for ablation comparison)
V0_OFFLOAD_LIKELIHOOD = np.array([
    [0.00, 0.15, 0.85],
    [0.00, 0.15, 0.85],
    [0.00, 0.15, 0.85],
])

LIKELIHOOD = {
    "FULL": np.array([
        [0.95, 0.04, 0.01],  # LOW:  1-3 streams → all clearly GOOD
        [0.25, 0.50, 0.25],  # MED:  n=4 borderline GOOD, n=5 MARGINAL
        [0.02, 0.13, 0.85],  # HIGH: n=6,7,8 all BAD (fps<10)
    ]),
    "LITE": np.array([
        [0.97, 0.03, 0.00],  # LOW:  1-3 streams → all clearly GOOD
        [0.80, 0.17, 0.03],  # MED:  n=4,5 still GOOD (fps>14)
        [0.05, 0.55, 0.40],  # HIGH: n=6,7 MARGINAL, n=8 BAD
    ]),
    "SKIP": np.array([
        [0.50, 0.50, 0.00],  # no inference → uninformative
        [0.50, 0.50, 0.00],
        [0.50, 0.50, 0.00],
    ]),
    # OFFLOAD: system-level observation after offloading one stream.
    # NOT the offloaded stream's own metrics (~7fps, BAD), but the
    # expected *system* observation: offloading frees local GPU,
    # so remaining streams improve.
    #
    # Design principle: OFFLOAD actions must be modeled in terms of
    # their effects on the residual local workload and global SLO,
    # rather than the offloaded stream's local QoS alone.
    #
    # At HIGH load: offloading has large marginal benefit (removing 1
    # of 6-8 streams significantly reduces GPU contention).
    # At LOW load: system already fine, offload is neutral/unnecessary.
    "OFFLOAD": np.array([
        [0.30, 0.55, 0.15],  # LOW:  system already fine, offload ≈ neutral
        [0.50, 0.35, 0.15],  # MED:  moderate relief, leans toward GOOD
        [0.60, 0.30, 0.10],  # HIGH: large marginal benefit, system improves
    ]),
}

# Transition model P(s' | s, a): how does load state change after action?
def transition_prob(s, a_type):
    """
    a_type: 'demote' (FULL->LITE, *->SKIP, *->OFFLOAD) decreases load
            'promote' (SKIP->LITE, LITE->FULL, OFFLOAD->LITE) increases load
            'noop': stays same
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


# ── AIF Controller ────────────────────────────────────────────────────────────

class AIFController(ControllerInterface):
    """
    Minimizes Expected Free Energy (EFE) at each step.

    EFE(a) = -E_Q[ln P(o|C)]           (pragmatic: expected preference)
             + E_Q[KL(Q(s|o,a)||Q(s|a))] (epistemic: information gain)

    Belief Q(s) is updated via Bayesian update after each observation.
    """

    def __init__(self, config_dir="/app/config",
                 offload_likelihood=None,
                 precision=4.0,
                 epistemic_weight=0.3,
                 cooldown=3):
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

        # OFFLOAD config: max offload = number of Nano URLs
        offload_cfg = action_cfg.get("modes", {}).get("OFFLOAD", {})
        self.offload_enabled = offload_cfg.get("enabled", False)
        offload_urls = offload_cfg.get("urls", [])
        self.max_offload = len(offload_urls) if self.offload_enabled else 0

        # Instance-level likelihood (supports ablation via constructor)
        self.likelihood = {k: v.copy() for k, v in LIKELIHOOD.items()}
        if offload_likelihood is not None:
            self.likelihood["OFFLOAD"] = np.array(offload_likelihood)

        # Preferred observation distribution C
        self.C = np.array([0.85, 0.12, 0.03])

        # Prior belief: uniform over load states
        self.belief = np.ones(N_LOAD_STATES) / N_LOAD_STATES

        self.modes = ["FULL", "LITE", "SKIP"]
        if self.offload_enabled:
            self.modes.append("OFFLOAD")

        # Hyperparameters (configurable for ablation)
        self.precision = precision
        self.epistemic_weight = epistemic_weight
        self.cooldown = cooldown

        # ASAP temporal smoothing
        self.asap_weight = 0.6

        # Anti-oscillation tracking
        self._last_switch = {}
        self._step = 0
        self._global_last_switch_step = -3
        self._prev_action_key = "noop"

    def on_episode_start(self, config):
        self.belief = np.ones(N_LOAD_STATES) / N_LOAD_STATES
        self._last_switch = {}
        self._step = 0
        self._global_last_switch_step = -3
        self._prev_action_key = "noop"

    def select_action(self, obs, stream_manager):
        per_stream = obs["per_stream"]
        n_active = obs["global"]["n_active_streams"]
        if n_active == 0:
            return None

        # 1. Update belief from current observation
        self.belief = self._update_belief(obs, per_stream)

        # 2. Get valid actions
        stream_ids = sorted(per_stream.keys())
        actions = self._enumerate_actions(per_stream, stream_ids,
                                          stream_manager, n_active)

        # 3. Compute EFE for each action
        efe_scores = {}
        for action_key, action in actions.items():
            efe = self._compute_efe(action_key, obs, per_stream, actions)
            efe_scores[action_key] = efe

        # 4. Precision-weighted softmax selection
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
        if steps_since_switch < self.cooldown:
            best_key = "noop"
        else:
            neg_efe = -self.precision * (scores - scores.min())
            probs   = np.exp(neg_efe)
            probs  /= probs.sum()
            best_key = np.random.choice(keys, p=probs)

        result = actions[best_key]
        self._prev_action_key = best_key

        # Track switch for anti-oscillation
        if result is not None:
            sid, new_mode = result
            old_mode = per_stream.get(sid, {}).get("current_mode", "FULL")
            self._last_switch[sid] = (old_mode, new_mode, self._step)
            self._global_last_switch_step = self._step
        self._step += 1

        return result

    def _update_belief(self, obs, per_stream):
        """Bayesian belief update: Q(s) ∝ P(o|s) * Q(s)"""
        # Get observation category from LOCAL inference streams only
        # SKIP and OFFLOAD are excluded (don't reflect local GPU state)
        obs_cats = []
        for sid, ps in per_stream.items():
            mode = ps.get("current_mode", "FULL")
            if mode in ("SKIP", "OFFLOAD"):
                continue
            fps = ps.get("infer_fps", ps.get("fps_avg", 0))
            lat = ps.get("infer_latency_p95_ms", ps.get("latency_p95_ms", 0))
            cat = self._categorize_obs(fps, lat)
            obs_cats.append(cat)

        if not obs_cats:
            return self.belief

        obs_cat = int(np.round(np.mean(obs_cats)))

        # Get representative mode for likelihood lookup
        modes_active = [ps.get("current_mode", "FULL")
                        for ps in per_stream.values()
                        if ps.get("current_mode") not in ("SKIP", "OFFLOAD")]
        rep_mode = modes_active[0] if modes_active else "FULL"

        # P(obs_cat | s) from likelihood table
        likelihood_vec = self.likelihood[rep_mode][:, obs_cat]

        # Bayesian update
        posterior = self.belief * likelihood_vec
        total = posterior.sum()
        if total > 1e-10:
            posterior /= total
        else:
            posterior = np.ones(N_LOAD_STATES) / N_LOAD_STATES

        return posterior

    def _categorize_obs(self, fps, lat):
        if fps >= 13 and lat <= 100:
            return OBS_GOOD
        elif fps >= self.fps_thresh and lat <= self.lat_thresh:
            return OBS_MARGINAL
        else:
            return OBS_BAD

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

    def _compute_efe(self, action_key, obs, per_stream, actions=None):
        """
        EFE(a) = pragmatic - 0.3 * epistemic + switch_penalty

        Pragmatic: -E_Q[ln C(o)] — how well does predicted obs match prefs
        Epistemic: E_Q[KL(Q(s|o,a)||Q(s|a))] — information gain
        """
        # Determine action type for transition model
        if action_key == "noop":
            a_type = "noop"
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
            a_type = self._action_type(old_mode, new_mode)

        # Predicted state distribution after action
        Q_s_given_a = np.zeros(N_LOAD_STATES)
        for s in range(N_LOAD_STATES):
            Q_s_given_a += self.belief[s] * transition_prob(s, a_type)

        # Likelihood for the target mode
        sim_mode = new_mode if new_mode else "FULL"
        likelihood_mat = self.likelihood[sim_mode]  # (N_states, N_obs)

        # Expected observation: P(o|a) = sum_s Q(s|a) P(o|s)
        P_o_given_a = Q_s_given_a @ likelihood_mat
        P_o_given_a = np.clip(P_o_given_a, 1e-10, 1.0)

        # Pragmatic value
        C_safe = np.clip(self.C, 1e-10, 1.0)
        pragmatic = -np.sum(P_o_given_a * np.log(C_safe))

        # Epistemic value
        epistemic = 0.0
        for o in range(N_OBS_CATS):
            if P_o_given_a[o] < 1e-10:
                continue
            Q_s_given_o_a = Q_s_given_a * likelihood_mat[:, o]
            norm = Q_s_given_o_a.sum()
            if norm > 1e-10:
                Q_s_given_o_a /= norm
            else:
                continue
            ratio = np.clip(Q_s_given_o_a / np.clip(Q_s_given_a, 1e-10, 1),
                            1e-10, 1e10)
            kl = np.sum(Q_s_given_o_a * np.log(ratio))
            epistemic += P_o_given_a[o] * kl

        # Switch cost penalty
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

        efe = pragmatic - self.epistemic_weight * epistemic + switch_penalty
        return efe

    def _action_type(self, old_mode, new_mode):
        rank = {"FULL": 2, "LITE": 1, "SKIP": 0, "OFFLOAD": 0}
        if rank.get(new_mode, 1) < rank.get(old_mode, 1):
            return "demote"
        elif rank.get(new_mode, 1) > rank.get(old_mode, 1):
            return "promote"
        return "noop"
