"""
Standard Discrete Active Inference Controller — 7-Module Architecture.

Implements a textbook-faithful AIF agent following the Process Theory
(Parr, Pezzulo, Friston 2022) and pymdp conventions, mapped onto the
multi-stream edge video scheduling problem.

====================================================================
Module Map (标准 AIF 七模块映射):
--------------------------------------------------------------------
  Module 1 — Generative Model (生成模型)
             A, B, C, D, E matrices encoding "how the agent believes
             the world works".

  Module 2 — Observation Input (观测输入)
             Multi-modal observations: O_local (stream quality) and
             O_system (GPU health).

  Module 3 — Hidden State Inference (隐藏状态推断)
             Bayesian model inversion: Q(s) ∝ P(o|s) · Q(s)_prior.
             Perception is interpretation, not raw reading.

  Module 4 — Policy Generation (策略生成)
             Enumerate candidate policies π = (u_0, u_1, ..., u_{T-1}).
             AIF compares multi-step strategies, not single actions.

  Module 5 — Policy Evaluation via EFE (策略评估)
             G(π) = Σ_τ [pragmatic(τ) + epistemic(τ)].
             Jointly optimizes goal-seeking and information-seeking.

  Module 6 — Policy Posterior & Action Execution (策略后验与动作执行)
             Q(π) ∝ E(π) · exp(-γ·G(π)).
             Action is a byproduct of policy inference:
             P(u_0) = Σ_{π: π[0]=u_0} Q(π).

  Module 7 — Learning & Model Update (学习与模型更新)
             Dirichlet-Categorical conjugate updates for A and B.
             The agent refines its world model through experience.
====================================================================

KNOWN LIMITATIONS (已知近似与设计折中):
--------------------------------------------------------------------
  L1 — ABSTRACT vs CONCRETE POLICY SEMANTICS
       Policies enumerate abstract action types {NOOP, DEMOTE, …},
       not concrete (stream_id, mode) pairs. EFE G(π) is evaluated
       against type-level transitions B1/B2 (expected average effect
       of each action type). Execution grounds each abstract action
       to a specific stream via _ground_action(), whose stream
       selection is heuristic and outside the generative model.
       → The AIF layer reasons over a type-level POMDP; execution
         implements a token-level MDP. B-matrix learning will absorb
         grounding variance as if it were transition stochasticity.
       Proper fix: expand state×action space to include per-stream
       identity, or learn stream selection jointly.

  L2 — FACTORIZED JOINT TRANSITION
       The joint rollout Q_joint_next = B1u @ Q_joint @ B2u.T
       assumes s1 and s2 transitions are action-conditionally
       independent. For the current B1/B2 this is exact; if an
       action couples both factors (e.g. OFFLOAD simultaneously
       reduces load AND increases offload level), a fully joint
       B[(s1',s2'),(s1,s2),u] would be needed.

  L3 — SOFT B-UPDATE CONFLATES DRIFT WITH TRANSITION
       Module 7 uses prev_qs × current_qs to update B counts,
       which mixes inference correction with genuine state change.
       Structurally acceptable for online prototyping; reviewers
       may flag this in a rigorous experimental setting.

  L4 — E(π) IS NOT LEARNED
       The policy prior E is set once per episode from action costs.
       It functions as an intervention-bias regularizer, not as a
       learned habit distribution in the canonical AIF sense.
====================================================================

State Factors (隐藏状态因子):
  s1: load_level   ∈ {LOW, MED, HIGH}     — local GPU contention
  s2: offload_level ∈ {NONE, PARTIAL, FULL} — offload utilization

Observation Modalities (观测模态):
  o1: stream_quality ∈ {GOOD, MARGINAL, BAD}
  o2: system_health  ∈ {GOOD, MARGINAL, BAD}

Control States (控制状态 / 抽象动作):
  u ∈ {NOOP, DEMOTE, PROMOTE, OFFLOAD, RECALL}

Policy Horizon (策略时域): T (configurable, default=3)
"""

import numpy as np
import yaml
from itertools import product

try:
    from engine.episode import ControllerInterface
except ImportError:
    # Fallback for testing outside Docker (no ultralytics/YOLO)
    class ControllerInterface:
        def select_action(self, observation, stream_manager):
            return None
        def on_episode_start(self, config):
            pass
        def on_episode_end(self, history):
            pass


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

# State factor 1: GPU load level
S_LOW, S_MED, S_HIGH = 0, 1, 2
N_S1 = 3  # |S_load|

# State factor 2: Offload utilization
OFF_NONE, OFF_PARTIAL, OFF_FULL = 0, 1, 2
N_S2 = 3  # |S_offload|

# Observation categories (shared by both modalities)
O_GOOD, O_MARG, O_BAD = 0, 1, 2
N_OBS = 3

# Control states (abstract actions)
U_NOOP     = 0  # Do nothing
U_DEMOTE   = 1  # Reduce one stream's quality (FULL→LITE or LITE→SKIP)
U_PROMOTE  = 2  # Increase one stream's quality (SKIP→LITE or LITE→FULL)
U_OFFLOAD  = 3  # Send one stream to a Nano worker
U_RECALL   = 4  # Bring back one offloaded stream to local
N_ACTIONS  = 5

ACTION_NAMES = ["NOOP", "DEMOTE", "PROMOTE", "OFFLOAD", "RECALL"]


# ═══════════════════════════════════════════════════════════════════
# MODULE 1: Generative Model (生成模型)
# ═══════════════════════════════════════════════════════════════════

def build_A1():
    """
    Observation model for modality 1: P(o_local | s_load).

    Maps hidden GPU load state to observable stream quality.
    Derived from hardware profiling on Jetson Orin Nano (YOLOv8n, 720p).

    Rows: load states (LOW, MED, HIGH)
    Cols: observation categories (GOOD, MARGINAL, BAD)
    """
    A1 = np.array([
        #  GOOD   MARG   BAD
        [0.90,  0.08,  0.02],   # LOW  — most streams meet SLO
        [0.30,  0.45,  0.25],   # MED  — near SLO boundary
        [0.03,  0.15,  0.82],   # HIGH — widespread violations
    ])
    return A1


def build_A2():
    """
    Observation model for modality 2: P(o_system | s_load, s_offload).

    Maps joint (load, offload) state to observable system health
    (derived from GPU utilization).

    Shape: (N_S2, N_S1, N_OBS) — indexed [s_offload, s_load, o_system].
    Key insight: offloading at HIGH load produces the largest improvement.
    """
    A2 = np.zeros((N_S2, N_S1, N_OBS))

    # No offload: system health mirrors load directly
    A2[OFF_NONE] = np.array([
        [0.88, 0.10, 0.02],   # LOW  — GPU healthy
        [0.15, 0.55, 0.30],   # MED  — moderate stress
        [0.02, 0.13, 0.85],   # HIGH — GPU saturated
    ])

    # Partial offload (1 stream): relieves some load
    A2[OFF_PARTIAL] = np.array([
        [0.90, 0.08, 0.02],   # LOW  — already fine
        [0.45, 0.40, 0.15],   # MED  — noticeable relief
        [0.15, 0.50, 0.35],   # HIGH — meaningful improvement
    ])

    # Full offload (2 streams): maximum relief
    A2[OFF_FULL] = np.array([
        [0.92, 0.06, 0.02],   # LOW  — marginal further benefit
        [0.65, 0.28, 0.07],   # MED  — strong improvement
        [0.30, 0.48, 0.22],   # HIGH — substantial relief
    ])

    return A2


def build_B1():
    """
    Transition model for factor 1: P(s_load' | s_load, u).

    Shape: (N_S1, N_S1, N_ACTIONS) — B1[:, :, u] is transition matrix
    for action u, where B1[s', s, u] = P(s_load'=s' | s_load=s, u).

    Semantics (ABSTRACT — see L1 in module docstring):
      These probabilities encode the *expected average* effect of each
      action type on the load state, marginalised over which specific
      stream will be selected by _ground_action(). The actual state
      change depends on the stream chosen and its current mode.

      DEMOTE/OFFLOAD → expected to reduce local GPU load → toward LOW
      PROMOTE/RECALL → expected to increase local GPU load → toward HIGH
      NOOP → mostly self-transition with slight drift
    """
    B1 = np.zeros((N_S1, N_S1, N_ACTIONS))

    for u in range(N_ACTIONS):
        for s in range(N_S1):
            if u in (U_DEMOTE, U_OFFLOAD):
                # Action reduces local load
                if s == S_LOW:
                    B1[:, s, u] = [1.0, 0.0, 0.0]  # already lowest
                elif s == S_MED:
                    B1[:, s, u] = [0.70, 0.25, 0.05]
                else:  # HIGH
                    B1[:, s, u] = [0.05, 0.70, 0.25]

            elif u in (U_PROMOTE, U_RECALL):
                # Action increases local load
                if s == S_HIGH:
                    B1[:, s, u] = [0.0, 0.0, 1.0]  # already highest
                elif s == S_MED:
                    B1[:, s, u] = [0.05, 0.25, 0.70]
                else:  # LOW
                    B1[:, s, u] = [0.25, 0.70, 0.05]

            else:  # NOOP
                B1[:, s, u] = [0.0, 0.0, 0.0]
                B1[s, s, u] = 0.90
                if s > 0:
                    B1[s - 1, s, u] = 0.05
                else:
                    B1[s, s, u] += 0.05
                if s < N_S1 - 1:
                    B1[s + 1, s, u] = 0.05
                else:
                    B1[s, s, u] += 0.05

    return B1


def build_B2():
    """
    Transition model for factor 2: P(s_offload' | s_offload, u).

    Shape: (N_S2, N_S2, N_ACTIONS).
    Only OFFLOAD and RECALL actions change offload state.
    """
    B2 = np.zeros((N_S2, N_S2, N_ACTIONS))

    for u in range(N_ACTIONS):
        for s in range(N_S2):
            if u == U_OFFLOAD:
                new_s = min(s + 1, N_S2 - 1)
                B2[:, s, u] = 0.0
                B2[new_s, s, u] = 0.90
                B2[s, s, u] += 0.10

            elif u == U_RECALL:
                new_s = max(s - 1, 0)
                B2[:, s, u] = 0.0
                B2[new_s, s, u] = 0.90
                B2[s, s, u] += 0.10

            else:  # NOOP, DEMOTE, PROMOTE — offload state unchanged
                B2[s, s, u] = 1.0

    return B2


def build_C():
    """
    Preference vectors: log P(o) encoding what the agent desires.

    C1: preference over stream quality (strongly prefer GOOD)
    C2: preference over system health (strongly prefer GOOD)

    In standard AIF, C is in log-space: C = ln(P_preferred(o)).
    """
    C1 = np.log(np.array([0.85, 0.12, 0.03]) + 1e-16)
    C2 = np.log(np.array([0.80, 0.15, 0.05]) + 1e-16)
    return C1, C2


def build_D():
    """
    Prior beliefs over initial hidden states (uniform = maximum ignorance).
    """
    D1 = np.ones(N_S1) / N_S1
    D2 = np.ones(N_S2) / N_S2
    return D1, D2


def build_E(policies, action_cost=None):
    """
    Policy prior E(π): encodes habitual preferences over policies.

    Policy prior E(π) encodes a prior over which policies are likely
    before EFE evidence is accumulated. Here E is shaped by action
    costs to create an *intervention bias*: NOOP is cheap, active
    interventions carry increasing cost. This functions as a
    regularizer that the EFE signal must overcome before the agent
    commits to an intervention.

    NOTE (see L4 in module docstring): this is NOT canonical habit
    learning — E is constructed once per episode from fixed costs and
    is never updated from experience. A learned E would require
    tracking policy posterior statistics across episodes and
    incrementally updating the Dirichlet concentration over policies.

    E(π) ∝ exp(-Σ_τ cost(u_τ))
    """
    if action_cost is None:
        # Default: NOOP is free, interventions have increasing cost
        action_cost = np.array([
            0.0,    # NOOP    — no cost (default habit)
            0.3,    # DEMOTE  — mild cost (reduces quality)
            0.4,    # PROMOTE — moderate cost (risks overload)
            0.6,    # OFFLOAD — higher cost (network, reduced FPS)
            0.5,    # RECALL  — moderate cost (increases local load)
        ])

    log_E = np.zeros(len(policies))
    for i, pi in enumerate(policies):
        log_E[i] = -sum(action_cost[u] for u in pi)

    E = np.exp(log_E)
    E /= E.sum()
    return E


# ═══════════════════════════════════════════════════════════════════
# MODULE 4: Policy Generation (策略生成)
# ═══════════════════════════════════════════════════════════════════

def construct_policies(T, n_actions=N_ACTIONS):
    """
    Enumerate all policies of horizon T.

    A policy π is a sequence of abstract actions: π = (u_0, u_1, ..., u_{T-1}).
    Total policies: |U|^T.

    With |U|=5, T=3: 125 policies. T=4: 625. T=5: 3125.
    All computationally feasible for our small state space.

    Returns: np.array of shape (n_policies, T), dtype=int.
    """
    policies = np.array(list(product(range(n_actions), repeat=T)), dtype=np.int32)
    return policies


# ═══════════════════════════════════════════════════════════════════
# THE CONTROLLER
# ═══════════════════════════════════════════════════════════════════

class StandardAIFController(ControllerInterface):
    """
    Standard 7-module discrete Active Inference controller.

    Key parameters:
        T          — policy horizon (planning depth, 策略时域)
        gamma      — EFE precision (策略精度, sharpness of policy posterior)
        lr_A       — learning rate for observation model (似然学习率)
        lr_B       — learning rate for transition model (转移学习率)
        use_learning — enable Module 7 online learning
    """

    def __init__(self, config_dir="/app/config",
                 T=3, gamma=4.0, lr_A=0.5, lr_B=0.2,
                 use_learning=False):
        self.config_dir = config_dir
        self.T = T
        self.gamma = gamma      # precision (精度) over policies
        self.lr_A = lr_A
        self.lr_B = lr_B
        self.use_learning = use_learning

        # ── Module 1: Generative Model ────────────────────────────
        self.A1 = build_A1()        # P(o_local | s_load)
        self.A2 = build_A2()        # P(o_system | s_load, s_offload)
        self.B1 = build_B1()        # P(s_load' | s_load, u)
        self.B2 = build_B2()        # P(s_offload' | s_offload, u)
        self.C1, self.C2 = build_C()  # log-preferences
        self.D1, self.D2 = build_D()  # initial state priors

        # Dirichlet concentration parameters (for Module 7 learning)
        # Initialize from the hand-specified model (informative prior)
        self._a1_counts = self.A1.copy() * 10 + 0.1  # prior strength=10
        self._a2_counts = self.A2.copy() * 10 + 0.1
        self._b1_counts = self.B1.copy() * 5 + 0.1
        self._b2_counts = self.B2.copy() * 5 + 0.1

        # E: policy prior (initially uniform, can be learned)
        # Constructed lazily after T is finalized
        self._policies = None
        self._E = None

        # ── Runtime state ─────────────────────────────────────────
        self.qs1 = self.D1.copy()   # Q(s_load)   — current belief
        self.qs2 = self.D2.copy()   # Q(s_offload) — current belief
        self.step = 0
        self.prev_action = None
        self.prev_obs = None

        # For compatibility with episode runner belief logging
        self.belief = self.qs1.copy()

        # Full joint posterior Q(s1,s2) — updated by _infer_states,
        # used to initialise policy rollout without re-factorising.
        self.Q_joint = np.outer(self.D1, self.D2)

        # Snapshots of beliefs at the moment an action was taken (for Module 7).
        # prev_Q_joint stores the full joint — marginals can be derived as:
        #   prev_qs1 = prev_Q_joint.sum(axis=1)
        #   prev_qs2 = prev_Q_joint.sum(axis=0)
        # Keeping the joint enables computing the full joint sufficient statistic
        # prev_Q_joint[s1_prev,s2_prev] * Q_joint[s1_next,s2_next] for future
        # fully-joint B-update, without re-running inference.
        self.prev_qs1 = None
        self.prev_qs2 = None
        self.prev_Q_joint = None   # full joint at action time; None until first step

        # ── Load configs ──────────────────────────────────────────
        self._load_configs()

    def _load_configs(self):
        try:
            with open(f"{self.config_dir}/slo.yaml") as f:
                self.slo_cfg = yaml.safe_load(f)
            with open(f"{self.config_dir}/switch_cost.yaml") as f:
                self.switch_cfg = yaml.safe_load(f)
            with open(f"{self.config_dir}/action_space.yaml") as f:
                self.action_cfg = yaml.safe_load(f)
        except FileNotFoundError:
            self.slo_cfg = {}
            self.switch_cfg = {}
            self.action_cfg = {}

        self.fps_thresh = self.slo_cfg.get("hard_constraints", {}).get(
            "min_fps", {}).get("threshold", 10)
        self.lat_thresh = self.slo_cfg.get("hard_constraints", {}).get(
            "p95_latency_ms", {}).get("threshold", 150)
        self.max_skip_ratio = self.slo_cfg.get("hard_constraints", {}).get(
            "max_skip_ratio", {}).get("threshold", 0.3)
        offload_cfg = self.action_cfg.get("modes", {}).get("OFFLOAD", {})
        self.max_offload = len(offload_cfg.get("urls", []))
        self.min_dwell = self.switch_cfg.get("min_dwell_before_exit",
                                              {"FULL": 1, "LITE": 1,
                                               "SKIP": 2, "OFFLOAD": 1})

    def on_episode_start(self, config):
        """Reset beliefs and step counter at episode start."""
        self.qs1 = self.D1.copy()
        self.qs2 = self.D2.copy()
        self.Q_joint = np.outer(self.D1, self.D2)
        self.belief = self.qs1.copy()
        self.step = 0
        self.prev_action = None
        self.prev_obs = None
        self.prev_qs1 = None
        self.prev_qs2 = None
        self.prev_Q_joint = None

        # Build policies for this episode
        self._policies = construct_policies(self.T)
        self._E = build_E(self._policies)

    # ===============================================================
    # MODULE 2: Observation Input (观测输入)
    # ===============================================================

    def _extract_observations(self, obs):
        """
        Extract two observation modalities from raw sensor data.

        O_local:  aggregate stream quality from FULL/LITE streams.
                  Agent sees outcomes, not causes — this is the core
                  partial observability.

        O_system: system-wide health from GPU utilization.

        Returns: (o1, o2) as integer category indices.
        """
        per_stream = obs.get("per_stream", {})
        gl = obs.get("global", {})

        # ── Modality 1: O_local ───────────────────────────────────
        obs_cats = []
        for sid, ps in per_stream.items():
            mode = ps.get("current_mode", "FULL")
            if mode in ("SKIP", "OFFLOAD"):
                continue  # excluded: don't reflect local GPU state
            fps = ps.get("infer_fps", ps.get("fps_avg", 0))
            lat = ps.get("infer_latency_p95_ms",
                         ps.get("latency_p95_ms", 0))
            obs_cats.append(self._categorize_stream(fps, lat))

        if obs_cats:
            o1 = int(round(np.mean(obs_cats)))
        else:
            o1 = O_MARG  # no local streams → uncertain

        # ── Modality 2: O_system ──────────────────────────────────
        gpu_util = gl.get("gpu_util_avg", 50.0)
        if gpu_util < 55:
            o2 = O_GOOD
        elif gpu_util < 78:
            o2 = O_MARG
        else:
            o2 = O_BAD

        return o1, o2

    def _categorize_stream(self, fps, lat):
        if fps >= 13 and lat <= 100:
            return O_GOOD
        elif fps >= self.fps_thresh and lat <= self.lat_thresh:
            return O_MARG
        return O_BAD

    # ===============================================================
    # MODULE 3: Hidden State Inference (隐藏状态推断)
    # ===============================================================

    def _infer_states(self, o1, o2):
        """
        Bayesian model inversion via exact joint posterior.

        Rather than sequential per-factor updates (which are an inconsistent
        approximation), we form the joint posterior Q(s1, s2) in one pass:

            Q(s1, s2) ∝ Q_prior(s1) · Q_prior(s2)
                        · P(o1 | s1)
                        · P(o2 | s1, s2)

        The prior factorizes as the product of current marginals (mean-field
        initialization). Because both observations are available simultaneously,
        no iteration is needed — the joint update is exact given this prior.
        Marginals qs1, qs2 are then derived by summing over the other factor.

        A2[s2, s1, o2] = P(o2 | s_offload=s2, s_load=s1), shape (N_S2, N_S1, N_OBS).
        L2_mat[s1, s2] = A2[s2, s1, o2] after transpose → shape (N_S1, N_S2).
        """
        # Joint prior from current factorized beliefs: (N_S1, N_S2)
        Q_joint = np.outer(self.qs1, self.qs2)

        # Likelihoods
        L1 = self.A1[:, o1]              # P(o1 | s1),       shape (N_S1,)
        L2_mat = self.A2[:, :, o2].T     # P(o2 | s1, s2),   shape (N_S1, N_S2)

        # Joint update: Q_joint[s1, s2] ∝ prior · L1[s1] · L2[s1, s2]
        Q_joint *= L1[:, np.newaxis]
        Q_joint *= L2_mat

        total = Q_joint.sum()
        if total < 1e-16:
            Q_joint = np.ones((N_S1, N_S2)) / (N_S1 * N_S2)
        else:
            Q_joint /= total

        # Marginals
        self.qs1 = Q_joint.sum(axis=1)   # Σ_{s2} Q(s1, s2)
        self.qs2 = Q_joint.sum(axis=0)   # Σ_{s1} Q(s1, s2)

        # Store full joint for policy rollout initialisation.
        # Using the true posterior (not outer(qs1,qs2)) preserves
        # cross-factor correlations induced by the joint observation.
        self.Q_joint = Q_joint

        # Update belief for episode runner logging
        self.belief = self.qs1.copy()

    # ===============================================================
    # MODULE 5: Policy Evaluation via EFE (策略评估)
    # ===============================================================

    def _evaluate_policy(self, policy):
        """
        Compute Expected Free Energy G(π) for a multi-step policy.

        G(π) = Σ_{τ=0}^{T-1} G_τ(π)

        where G_τ = pragmatic_τ - epistemic_τ:

        pragmatic_τ = -E_{Q(o|π)}[ln C(o)]
            → "Will future observations match my preferences?"

        epistemic_τ = E_{Q(o1,o2|π)}[KL[Q(s1,s2|o1,o2,π) || Q(s1,s2|π)]]
            → "Will this policy help me learn about the JOINT hidden state?"

        Joint rollout:
            Q_joint_next = B1u @ Q_joint @ B2u.T
        Propagates the full joint distribution; preserves cross-factor
        correlations rather than collapsing to factorised marginals.

        Joint epistemic:
            L[s1,s2] = P(o1|s1)·P(o2|s1,s2)  over all (o1,o2) pairs.
        Covers cross-factor information that per-factor sums IG(o1;s1)
        + IG(o2;s2) would miss.

        Returns: scalar G(π) (lower = better policy).
        """
        G_total = 0.0

        # Start rollout from true joint posterior (stored by _infer_states).
        # Avoids re-factorising to outer(qs1, qs2), which would discard
        # cross-factor correlations induced by the joint observation.
        Q_joint_pred = self.Q_joint.copy()   # (N_S1, N_S2)

        for tau in range(len(policy)):
            u = policy[tau]

            # --- Joint state prediction ---
            # Q(s1',s2') = Σ_{s1,s2} P(s1'|s1,u)·P(s2'|s2,u)·Q(s1,s2)
            #            = B1u @ Q_joint @ B2u.T
            B1u = self.B1[:, :, u]                        # (N_S1, N_S1)
            B2u = self.B2[:, :, u]                        # (N_S2, N_S2)
            Q_joint_next = B1u @ Q_joint_pred @ B2u.T     # (N_S1, N_S2)
            total = Q_joint_next.sum()
            Q_joint_next = (Q_joint_next / total if total > 1e-16
                            else np.ones((N_S1, N_S2)) / (N_S1 * N_S2))

            # Marginals (for pragmatic term and diagnostics)
            qs1_next = Q_joint_next.sum(axis=1)            # (N_S1,)

            # --- Predict future observations ---
            qo1 = self.A1.T @ qs1_next                    # (N_OBS,)
            qo1 = np.clip(qo1, 1e-16, None)

            # Q(o2|π) = Σ_{s1,s2} Q_joint[s1,s2]·P(o2|s1,s2)
            # A2 shape (N_S2,N_S1,N_OBS); Q_joint_next.T shape (N_S2,N_S1)
            qo2 = (self.A2 * Q_joint_next.T[:, :, np.newaxis]).sum(axis=(0, 1))
            qo2 = np.clip(qo2, 1e-16, None)

            # --- Pragmatic value ---
            pragmatic = -np.dot(qo1, self.C1) - np.dot(qo2, self.C2)

            # --- Fully joint epistemic value ---
            # E_{Q(o1,o2)}[KL[Q(s1,s2|o1,o2,π) || Q(s1,s2|π)]]
            # L[s1,s2] = P(o1|s1)·P(o2|s1,s2)  — joint observation likelihood
            # Q_post = Q_joint_next * L  (unnormalised posterior)
            # p_obs  = sum(Q_post)       = P(o1,o2|π)
            epistemic = 0.0
            for o1_val in range(N_OBS):
                A1_col = self.A1[:, o1_val]               # (N_S1,)
                for o2_val in range(N_OBS):
                    # A2[:,:,o2].T[s1,s2] = P(o2|s_load=s1, s_off=s2)
                    A2_slice = self.A2[:, :, o2_val].T    # (N_S1, N_S2)
                    L = A1_col[:, np.newaxis] * A2_slice  # (N_S1, N_S2)

                    Q_post = Q_joint_next * L
                    p_obs = Q_post.sum()
                    if p_obs < 1e-16:
                        continue
                    Q_post /= p_obs

                    kl = np.sum(Q_post * np.log(
                        np.clip(Q_post, 1e-16, None) /
                        np.clip(Q_joint_next, 1e-16, None)
                    ))
                    epistemic += p_obs * kl

            # epistemic subtracted: information-seeking reduces G
            G_total += pragmatic - epistemic
            Q_joint_pred = Q_joint_next

        return G_total

    # ===============================================================
    # MODULE 6: Policy Posterior & Action Execution
    #           (策略后验与动作执行)
    # ===============================================================

    def _select_action(self):
        """
        Form policy posterior, marginalize to abstract action.

        Q(π) ∝ E(π) · exp(-γ · G(π))
        P(u_0) = Σ_{π: π[0]=u_0} Q(π)

        The action is a BYPRODUCT of policy inference —
        not a direct optimization over single actions.
        Concrete grounding (which stream?) happens separately in _ground_action.
        """
        policies = self._policies
        n_policies = len(policies)

        # --- Evaluate all policies ---
        G = np.zeros(n_policies)
        for i in range(n_policies):
            G[i] = self._evaluate_policy(policies[i])

        # --- Policy posterior ---
        # Q(π) ∝ E(π) · exp(-γ · G(π))
        log_posterior = np.log(self._E + 1e-16) - self.gamma * G
        log_posterior -= log_posterior.max()  # numerical stability
        q_pi = np.exp(log_posterior)
        q_pi /= q_pi.sum()

        # --- Marginalize to first action ---
        # P(u_0) = Σ_{π: π[0]=u_0} Q(π)
        p_u0 = np.zeros(N_ACTIONS)
        for i in range(n_policies):
            p_u0[policies[i, 0]] += q_pi[i]
        p_u0 = self._normalize(p_u0)

        # --- Select abstract action ---
        u_selected = np.random.choice(N_ACTIONS, p=p_u0)

        # Store diagnostics
        self._last_G = G
        self._last_q_pi = q_pi
        self._last_p_u0 = p_u0

        return u_selected

    # ===============================================================
    # MODULE 7: Learning & Model Update (学习与模型更新)
    # ===============================================================

    def _learning_update(self, o1, o2, u_prev):
        """
        Dirichlet-Categorical conjugate update for observation
        and transition models.

        After observing (o1, o2) and having taken action u_prev:
        - Update A counts: "in state s, I observed o with this frequency"
        - Update B counts: "after action u in state s, I transitioned to s'"

        This is how the agent LEARNS its world model through experience,
        rather than relying on hand-specified matrices.
        """
        if not self.use_learning:
            return

        # --- Update A1 (observation model for local quality) ---
        # Weight the update by current belief (soft assignment)
        for s1 in range(N_S1):
            self._a1_counts[s1, o1] += self.lr_A * self.qs1[s1]
        # Re-derive A1 from counts
        self.A1 = self._a1_counts / self._a1_counts.sum(
            axis=1, keepdims=True)

        # --- Update A2 (observation model for system health) ---
        # Use the true joint posterior Q(s1,s2) rather than the factorized
        # outer product qs1⊗qs2: A2's likelihood P(o2|s1,s2) depends on BOTH
        # factors, so the correct soft-assignment weight is the joint belief,
        # not a product of marginals (which discards cross-factor correlation).
        for s2 in range(N_S2):
            for s1 in range(N_S1):
                weight = self.Q_joint[s1, s2]
                self._a2_counts[s2, s1, o2] += self.lr_A * weight
        # Re-derive A2 from counts
        for s2 in range(N_S2):
            row_sums = self._a2_counts[s2].sum(axis=1, keepdims=True)
            self.A2[s2] = self._a2_counts[s2] / np.clip(row_sums, 1e-16, None)

        # --- Update B1 (transition model for load) ---
        # NOTE: soft assignment uses prev_qs1 (belief at action time) × qs1
        # (belief after transition). This approximates the true sufficient
        # statistic but conflates belief drift with genuine state change.
        # Acceptable for online prototype; a cleaner approach would use the
        # B-predicted prior Q(s'|π) as the target rather than the posterior.
        if u_prev is not None and self.prev_Q_joint is not None:
            # Derive marginals from the stored joint (mathematically equivalent
            # to prev_qs1/prev_qs2, but makes the source of truth explicit).
            # Future joint-B learning would use prev_Q_joint[s1_p,s2_p] *
            # Q_joint[s1_n,s2_n] as the sufficient statistic directly.
            prev_qs1_j = self.prev_Q_joint.sum(axis=1)   # (N_S1,)
            for s_prev in range(N_S1):
                for s_next in range(N_S1):
                    weight = prev_qs1_j[s_prev] * self.qs1[s_next]
                    self._b1_counts[s_next, s_prev, u_prev] += (
                        self.lr_B * weight)
            # Re-derive B1
            for u in range(N_ACTIONS):
                col_sums = self._b1_counts[:, :, u].sum(axis=0, keepdims=True)
                self.B1[:, :, u] = self._b1_counts[:, :, u] / np.clip(
                    col_sums, 1e-16, None)

        # --- Update B2 (transition model for offload state) ---
        if u_prev is not None and self.prev_Q_joint is not None:
            prev_qs2_j = self.prev_Q_joint.sum(axis=0)   # (N_S2,)
            for s_prev in range(N_S2):
                for s_next in range(N_S2):
                    weight = prev_qs2_j[s_prev] * self.qs2[s_next]
                    self._b2_counts[s_next, s_prev, u_prev] += (
                        self.lr_B * weight)
            # Re-derive B2
            for u in range(N_ACTIONS):
                col_sums = self._b2_counts[:, :, u].sum(axis=0, keepdims=True)
                self.B2[:, :, u] = self._b2_counts[:, :, u] / np.clip(
                    col_sums, 1e-16, None)

    # ===============================================================
    # Action Grounding: Abstract → Concrete (抽象动作 → 具体执行)
    # ===============================================================

    def _ground_action(self, u, obs, stream_manager):
        """
        Map abstract control state u to concrete (stream_id, new_mode).

        KNOWN LIMITATION — semantic gap between AIF policy and execution:
        The AIF generative model reasons over abstract action types
        (DEMOTE, OFFLOAD, …) and their expected effects on load/offload
        states. But the physical system needs a specific (stream_id, mode).
        This grounding step is a heuristic ("pick worst/best/heaviest
        stream") that is NOT part of the AIF inference — it lives outside
        the generative model.

        Consequence: the policy EFE G(π) is evaluated against abstract
        action semantics, but the actual state change depends on which
        stream is selected here. If the heuristic systematically deviates
        from the assumed action effects in B1/B2, the generative model's
        predictions will be biased, and Module 7 learning will absorb
        that bias into the transition counts.

        A proper fix requires expanding the state/action space to include
        per-stream identity, or learning a stream-selection policy jointly
        with the AIF inference.
        """
        per_stream = obs.get("per_stream", {})
        if not per_stream:
            return None

        sm = stream_manager
        active_ids = list(per_stream.keys())

        if u == U_NOOP:
            return None

        elif u == U_DEMOTE:
            # Find worst-performing local stream, demote one step
            worst_sid, worst_score = None, float("inf")
            for sid in active_ids:
                ps = per_stream[sid]
                mode = ps.get("current_mode", "FULL")
                if mode in ("SKIP",):
                    continue  # already at bottom
                dwell = sm.get_mode_dwell(sid)
                if dwell < self.min_dwell.get(mode, 1):
                    continue
                fps = ps.get("infer_fps", ps.get("fps_avg", 30))
                lat = ps.get("infer_latency_p95_ms",
                             ps.get("latency_p95_ms", 0))
                score = fps - lat / 30.0  # lower = worse
                if score < worst_score:
                    worst_score = score
                    worst_sid = sid
            if worst_sid is None:
                return None
            old_mode = per_stream[worst_sid].get("current_mode", "FULL")
            new_mode = {"FULL": "LITE", "LITE": "SKIP",
                        "OFFLOAD": "SKIP"}.get(old_mode)
            if new_mode and self._check_constraints(new_mode, per_stream):
                return (worst_sid, new_mode)
            return None

        elif u == U_PROMOTE:
            # Find best candidate in degraded mode, promote one step
            best_sid, best_score = None, -float("inf")
            for sid in active_ids:
                ps = per_stream[sid]
                mode = ps.get("current_mode", "FULL")
                if mode == "FULL":
                    continue  # already at top
                dwell = sm.get_mode_dwell(sid)
                if dwell < self.min_dwell.get(mode, 1):
                    continue
                fps = ps.get("infer_fps", ps.get("fps_avg", 0))
                score = -fps  # promote the one with worst fps (most to gain)
                if mode in ("SKIP",):
                    score -= 100  # prioritize un-skipping
                if score > best_score:
                    best_score = score
                    best_sid = sid
            if best_sid is None:
                return None
            old_mode = per_stream[best_sid].get("current_mode", "SKIP")
            new_mode = {"SKIP": "LITE", "LITE": "FULL"}.get(old_mode)
            if new_mode:
                return (best_sid, new_mode)
            return None

        elif u == U_OFFLOAD:
            # Find heaviest local stream to offload
            n_offload = sum(1 for ps in per_stream.values()
                           if ps.get("current_mode") == "OFFLOAD")
            if n_offload >= self.max_offload:
                return None
            best_sid = None
            best_load = -1
            for sid in active_ids:
                ps = per_stream[sid]
                mode = ps.get("current_mode", "FULL")
                if mode in ("SKIP", "OFFLOAD"):
                    continue
                dwell = sm.get_mode_dwell(sid)
                if dwell < self.min_dwell.get(mode, 1):
                    continue
                lat = ps.get("infer_latency_p95_ms",
                             ps.get("latency_p95_ms", 0))
                if lat > best_load:
                    best_load = lat
                    best_sid = sid
            if best_sid:
                return (best_sid, "OFFLOAD")
            return None

        elif u == U_RECALL:
            # Bring back most recently offloaded stream
            for sid in active_ids:
                ps = per_stream[sid]
                if ps.get("current_mode") == "OFFLOAD":
                    dwell = sm.get_mode_dwell(sid)
                    if dwell >= self.min_dwell.get("OFFLOAD", 1):
                        return (sid, "LITE")
            return None

        return None

    def _check_constraints(self, new_mode, per_stream):
        """Check if adding a stream in new_mode violates constraints."""
        if new_mode == "SKIP":
            n_skip = sum(1 for ps in per_stream.values()
                        if ps.get("current_mode") == "SKIP")
            n_active = len(per_stream)
            if n_active > 0 and (n_skip + 1) / n_active > self.max_skip_ratio:
                return False
        return True

    # ===============================================================
    # Main Entry Point
    # ===============================================================

    def select_action(self, observation, stream_manager):
        """
        Standard AIF control loop — one cycle through all 7 modules.

        Timeline per step:
          1. Receive observations         (Module 2)
          2. Update beliefs               (Module 3)
          3. [Generate policies]           (Module 4, done once)
          4. Evaluate all policies via EFE (Module 5)
          5. Form policy posterior         (Module 6)
          6. Marginalize → abstract action (Module 6)
          7. Ground to concrete action     (Action Grounding)
          8. Learn from experience         (Module 7)
        """
        # Snapshot beliefs *before* this step's inference (Module 7 needs them).
        # Store the full joint so _learning_update can derive marginals from it
        # and future analyses can compare factorized vs joint sufficient stats.
        self.prev_qs1 = self.qs1.copy()
        self.prev_qs2 = self.qs2.copy()
        self.prev_Q_joint = self.Q_joint.copy()

        # ── Module 2: Observation Input ───────────────────────────
        o1, o2 = self._extract_observations(observation)

        # ── Module 3: Hidden State Inference ──────────────────────
        self._infer_states(o1, o2)

        # ── Module 7: Learning (uses previous action + new obs) ───
        u_prev = self.prev_action
        self._learning_update(o1, o2, u_prev)

        # ── Modules 4-6: Policy Generation → Evaluation → Action ─
        if self._policies is None:
            self._policies = construct_policies(self.T)
            self._E = build_E(self._policies)

        u_abstract = self._select_action()

        # ── Action Grounding ──────────────────────────────────────
        concrete_action = self._ground_action(
            u_abstract, observation, stream_manager)

        # Cache for semantic-gap analysis in get_diagnostics()
        self._last_u_abstract = u_abstract
        self._last_concrete_action = concrete_action  # (sid, mode) or None

        # Track for next step
        self.prev_action = u_abstract
        self.prev_obs = (o1, o2)
        self.step += 1

        return concrete_action

    # ===============================================================
    # Utilities
    # ===============================================================

    @staticmethod
    def _normalize(x):
        """Normalize a distribution, handling edge cases."""
        total = x.sum()
        if total < 1e-16:
            return np.ones_like(x) / len(x)
        return x / total

    @staticmethod
    def _kl_divergence(p, q):
        """KL[p || q] with numerical safety."""
        p_safe = np.clip(p, 1e-16, None)
        q_safe = np.clip(q, 1e-16, None)
        return np.sum(p_safe * np.log(p_safe / q_safe))

    # ===============================================================
    # Diagnostics (for analysis and plotting)
    # ===============================================================

    def get_diagnostics(self):
        """Return internal state for interpretability analysis."""
        diag = {
            "belief_load":    self.qs1.tolist(),
            "belief_offload": self.qs2.tolist(),
            "belief_joint":   self.Q_joint.tolist(),  # full joint Q(s1,s2)
            "step": self.step,
        }
        # Semantic-gap tracing: abstract policy choice vs concrete execution.
        # Null concrete_action means grounding found no eligible stream —
        # useful for detecting over-conservative constraint violations.
        if hasattr(self, "_last_u_abstract"):
            diag["u_abstract"]      = ACTION_NAMES[self._last_u_abstract]
            diag["concrete_action"] = self._last_concrete_action  # (sid, mode) | None
            diag["grounding_null"]  = self._last_concrete_action is None
        if hasattr(self, "_last_p_u0"):
            diag["action_probs"] = {
                ACTION_NAMES[i]: float(self._last_p_u0[i])
                for i in range(N_ACTIONS)
            }
        if hasattr(self, "_last_G"):
            # Top-5 policies by posterior probability
            if hasattr(self, "_last_q_pi"):
                top5_idx = np.argsort(self._last_q_pi)[-5:][::-1]
                diag["top_policies"] = [
                    {
                        "policy": [ACTION_NAMES[u]
                                   for u in self._policies[i]],
                        "G": float(self._last_G[i]),
                        "posterior": float(self._last_q_pi[i]),
                    }
                    for i in top5_idx
                ]
        return diag
