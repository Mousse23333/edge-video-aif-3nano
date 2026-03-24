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

So I'm working on this: we have a three-node **Jetson Orin Nano** cluster. Three identical devices — one acts as the coordinator, running the control loop and doing local GPU inference, and the other two are just stateless offload workers that take jobs over HTTP.

Now, when you run multiple AI video streams concurrently on this thing, you hit GPU contention. And it's not subtle — with YOLOv8n on 720p video, adding a sixth stream drops per-stream throughput from 10.6 down to 8.7 FPS. That's below our 10 FPS SLO. P95 latency jumps from 92 to 107 ms.

So you need a scheduler. One that, every second, looks at each stream and decides: should this run locally at full resolution (FULL, 640px), drop to a lighter model (LITE, 320px), skip this frame entirely (SKIP), or ship it off to a worker node (OFFLOAD)?

And importantly — this is not a simulation. We run this on real hardware, compare four controllers, across four workload patterns, five independent runs each.

### 2. Why Is This Hard

The obvious answer is "the action space is large." But that's not really the hard part.

The hard part is that **you can't see the true system state**. What the scheduler sees is FPS, P95 latency, GPU utilization — all noisy, all lagged. What actually drives performance is the underlying GPU contention level, thermal throttling, queueing effects. All hidden.

So this isn't a threshold problem — "if FPS drops below X, do Y." That framing assumes what you observe is ground truth. It isn't. The real structure here is a **POMDP**: you have to infer what's actually going on inside the system, and then decide how to act.

### 3. The Core Idea

My core idea isn't "use a fancier algorithm." It's simpler than that:

> Don't just react to observations. Maintain an explicit **probabilistic belief** about the hidden load state, and make sure the **system-level consequences** of each action are correctly encoded in the model.

That second part is where everything lives. Two actions in particular:

- **OFFLOAD**: If you encode it naïvely — using the offloaded stream's own performance (~7 FPS, near-BAD) — the controller concludes OFFLOAD is a bad action and never uses it. But that's wrong. The value of OFFLOAD isn't that the offloaded stream performs better. It's that you free up local GPU capacity, which benefits all the remaining streams. That system-level effect has to be explicitly written into the model.
- **SKIP**: If SKIP is encoded as a neutral observation — "skipping produces no signal" — then from the controller's perspective, SKIP is free. It'll overuse it constantly. Once you encode SKIP as "this is degraded service, there's a real cost," that behavior disappears on its own.

### 4. How It Works

Concretely, here's the setup:

1. The hidden system state is discretized into three levels: **LOW** (≤3 streams inferring locally), **MEDIUM** (4–5), **HIGH** (≥6).
2. Every 1-second control step, we do a **Bayesian belief update** using the latest noisy observation.
3. For each candidate action, we compute the **Expected Free Energy (EFE)** — which has two terms. The pragmatic term asks: how well does the predicted outcome match our goal? The epistemic term asks: how much would this action tell us about the hidden state? That two-term structure is what separates AIF from just doing Bayesian filtering with a utility function on top.
4. We pick the action with the lowest EFE, using a precision-weighted softmax.

One thing worth flagging: this is a single-step approximation — T=1. We're not doing the full multi-step policy inference that canonical AIF does. That's a deliberate engineering call. Real-time edge control, 1-second intervals, under 1 ms per step on Jetson hardware. The belief state still carries forward history, so we're not fully memoryless — but we are acting greedily at each step.

The key isn't the math. It's what you write into the model for "what does this action do to the overall system."

### 5. Why It Matters

Here's what the experiments show:

- Default AIF at 86.0% SLO is basically on par with the threshold heuristic at 87.8%. Respectable, but not impressive.
- DQN hits 94.9% in steady overload, so it can do better — but in the burst scenario, its behavior is all over the place. SLO ranges from 64.6% to 94.3% across seeds. Switch counts from 2 to 87. Five episodes of training just isn't enough to produce a reliable policy.
- **But here's the key result: once we fix the OFFLOAD and SKIP likelihoods to use system-level semantics, AIF goes from 86% to 93.2%. That beats the myopic greedy baseline at 89.9% — which has access to profiling data.**

And the ablation hierarchy makes the real lesson concrete: those two likelihood corrections together contribute about 16 percentage points. Everything else — precision, epistemic weight, cooldown — combined gives you less than 8 pp.

Model specification is the dominant factor. Not the algorithm.

---

## One-Sentence Core

> The question this work addresses is not whether AIF can be applied to edge scheduling. It is: **when the environment is partially observable, does explicit hidden-state inference and system-level action modeling actually change scheduling behavior — and by how much?**
