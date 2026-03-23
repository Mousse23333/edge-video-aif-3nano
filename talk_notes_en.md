# Research Overview: Active Inference for Multi-Stream Edge Video Scheduling

## Top-Level Question

> **When the true GPU load state is not directly observable, should a scheduler react to noisy observations directly — or first infer the hidden state, then act?**

---

## Three Key Points

**Key 1: Edge scheduling is a POMDP, not a threshold problem.**
What the scheduler observes — FPS, latency, GPU utilization — is noisy and lagged. The true GPU contention state is a hidden variable. Decisions must be made under partial observability, which no threshold-based rule can fully address.

**Key 2: Action semantics must be system-level, not per-stream.**
OFFLOAD and SKIP cannot be encoded only through the affected stream's own QoS. OFFLOAD's value is not that the offloaded stream performs better — it is that offloading relieves local GPU pressure, improving the remaining streams. SKIP is not a free action; it is a degraded-service choice with real coverage cost. Encoding these correctly in the generative model is what enables the right decisions.

**Key 3: Structured control vs. trained control — two fundamentally different engineering trade-offs.**
AIF requires no training, works from step one, and produces low cross-run variance (std ≤ 4.5%). DQN can achieve higher peak SLO, but exhibits high seed-dependent variance (burst scenario: 64%–94% across seeds, switch counts ranging from 2 to 87). For deployments where predictability matters, model-driven control is a better fit.

---

## Detailed Narrative

### 1. What Is the Problem

I am working on the following problem: on a three-node **Jetson Orin Nano** cluster — three identical nodes where one acts as the coordinator running the control loop and local inference, and two act as stateless offload workers — running multiple concurrent AI video inference streams causes GPU resource contention that degrades per-stream FPS and increases latency as stream count grows. With YOLOv8n on 720p video, adding a sixth stream drops per-stream throughput from 10.6 to 8.7 FPS — below the 10 FPS SLO — while P95 latency rises from 92 to 107 ms.

This requires a scheduler that dynamically decides, for each stream, how it should be processed: local inference at full resolution (FULL, 640px), local inference at reduced resolution (LITE, 320px), skipped entirely (SKIP), or offloaded via HTTP to a worker node (OFFLOAD).

This is not a simulation study. We evaluate four controllers — Heuristic, Myopic Greedy, DQN, and AIF — on a real three-node Jetson testbed, across four workload scenarios (Ramp-up, Burst, Steady Overload, Oscillating), with five independent runs per controller–scenario pair.

### 2. Why Is This Hard

The difficulty is not just the large action space. The fundamental challenge is that **the true system state cannot be directly observed**.

The scheduler sees only noisy, lagged signals: inference FPS, P95 latency, GPU utilization. But what actually determines per-stream performance is the underlying GPU contention level, thermal throttling, and queueing state — all of which are hidden.

This means the problem is not a simple threshold-triggering problem. It is a **POMDP**: the scheduler must infer the hidden system load state from noisy observations, then decide how to act. Any approach that treats observations as ground truth will systematically misread the system state.

### 3. The Core Idea

The core insight is not "use a more complex algorithm." It is:

> A scheduler should not react to observations directly. It should explicitly maintain a **probabilistic belief** over the hidden load state, and encode the **system-level semantics** of each action in the generative model.

This matters most for two actions:

- **OFFLOAD**: If encoded using the offloaded stream's own QoS (~7 FPS, near-BAD), the controller treats OFFLOAD as a "bad action" and never uses it. But the true value of OFFLOAD is that it frees local GPU capacity, improving the remaining local streams. This system-level effect must be explicitly encoded.
- **SKIP**: If encoded as a neutral observation ("no signal"), the policy treats SKIP as a free action and overuses it. Once encoded as "frame-skipping has a cost — it is degraded service," excessive SKIP behavior disappears naturally.

### 4. How It Works

The concrete implementation:

1. The hidden system state is abstracted into three discrete load levels: **LOW** (≤3 local inference streams), **MEDIUM** (4–5), **HIGH** (≥6).
2. At each 1-second control step, the controller performs a **Bayesian belief update** given the current noisy observation.
3. For each candidate action, the controller computes the **Expected Free Energy (EFE)** — roughly, how well does the predicted post-action observation match the target preference distribution?
4. Actions are selected via a precision-weighted softmax over EFE values.

The key is not the equations themselves, but how the generative model defines "what consequence does this action have on the overall system state." The entire per-step computation takes less than 1 ms — fully compatible with real-time control.

### 5. Why It Matters

The experimental results show a clear pattern:

- Default AIF (86.0% SLO) is on par with the Heuristic (87.8%).
- DQN peaks at 94.9% in some runs, but exhibits high seed-dependent variance in the burst scenario (range: 64.6%–94.3%; switch counts from 2 to 87) — indicating that short-episode training produces fundamentally unpredictable policies.
- **Once OFFLOAD and SKIP likelihoods are corrected from per-stream to system-level semantics, AIF's average SLO improves from 86% to 93.2%, surpassing the profiling-based myopic greedy baseline (89.9%).**

The deeper conclusion is that for this class of edge scheduling problems, **how actions are encoded in the generative model is a core part of the method** — not an implementation detail. The ablation hierarchy makes this concrete: the two likelihood corrections together account for ~16 pp of improvement, while all algorithmic parameters (precision, epistemic weight, cooldown) combined contribute less than 8 pp.

Model specification, not algorithmic complexity, is the primary determinant of performance.

---

## One-Sentence Core

> The question this work addresses is not whether AIF can be applied to edge scheduling. It is: **when the environment is partially observable, does explicit hidden-state inference and system-level action modeling actually change scheduling behavior — and by how much?**
