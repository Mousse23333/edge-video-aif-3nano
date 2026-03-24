# 工作介绍：基于 Active Inference 的边缘多流视频调度

## 顶层问题

> **当边缘 GPU 的真实负载状态不可见时，调度器应该直接响应 noisy observations，还是应该先推断隐藏状态再行动？**

---

## 三个关键点

**Key 1：Edge scheduling is a POMDP, not a threshold problem.**
调度器能观测到的只是带噪声、带滞后的 FPS / latency / GPU 利用率，真实的 GPU 竞争状态是隐藏变量，必须在部分可观测条件下做决策。

**Key 2：Action semantics must be system-level.**
OFFLOAD 和 SKIP 不能只按单流 QoS 编码。OFFLOAD 的价值不在于被迁出那一路流本身变好，而在于它释放了本地 GPU 压力；SKIP 不是无损动作，而是有代价的降级。一旦生成模型的语义写错，策略会系统性误判动作价值。

**Key 3：Structured control vs. trained control，两种不同的工程 trade-off。**
AIF 不需要训练，从第一步就可用，跨运行方差低（std ≤ 4.5%）；DQN 峰值更高，但跨 seed 行为不稳定（burst 场景 SLO 在 64%–94% 之间波动）。对部署时有可预测性要求的场景，模型驱动控制器更合适。

---

## 详细口述稿

### 1. 问题是什么

我在研究的是这样一个问题：在一个三节点的 **Jetson Orin Nano** 集群上——三台硬件相同，一台作为 coordinator 跑控制器和本地推理，另外两台作为 offload worker——同时跑多路视频 AI 推理时，随着流数上升，GPU 资源竞争会显著拉低每路流的 FPS 并推高 latency。以 YOLOv8n 为例，6 路流时帧率从 10.6 掉到 8.7 FPS，直接触发 SLO 违规。

因此需要一个调度器，动态决定每路流该如何处理——本地 FULL（640px）、本地 LITE（320px）、SKIP，或者 OFFLOAD 到 worker 节点。

这个问题不是纯模拟，我们在真实三节点硬件上比较了四种控制器：Heuristic、Myopic Greedy、DQN 和 AIF，用四种工作负载场景（Ramp-up、Burst、Steady Overload、Oscillating）进行评估，每种配置独立运行 5 次。

### 2. 为什么 Hard

这个问题难，不只是因为动作空间大，而是因为**真实系统状态不可直接观测**。

调度器能看到的只是带噪声、带滞后的指标：infer FPS、P95 latency、GPU 利用率。但真正决定每路流性能的是 GPU 竞争状态、热功耗节流、排队效应——这些都是隐藏的。

所以这个问题本质上不是阈值触发问题，而是一个 **POMDP**：必须在部分可观测条件下，从 noisy observations 里推断系统处于什么负载状态，再决定动作。

### 3. 核心想法

我的核心想法不是"换一个更复杂的算法"，而是：

> 调度器不应该只对观测直接反应，而应该显式维护对隐藏负载状态的 **belief**，并且把动作的**系统级语义**写进生成模型。

这件事在两个动作上特别明显：

- **OFFLOAD**：如果只按被迁出那路流自己的 QoS 编码（~7 FPS，near-BAD），控制器会把 OFFLOAD 当成"坏动作"永远不用。但 OFFLOAD 真正的价值是释放本地 GPU 压力，改善剩余流的整体状态——这个系统级效果必须显式编码。
- **SKIP**：如果被写成中性观测（"跳帧不产生任何信号"），策略会把 SKIP 当成无损选项滥用。一旦改写成"跳帧是有代价的降级服务"，过度 SKIP 的问题就自然消失。

### 4. 它是如何工作的

具体做法：

1. 把系统隐藏状态抽象为三个离散负载等级：**LOW**（≤3 路本地推理）、**MEDIUM**（4–5 路）、**HIGH**（≥6 路）。
2. 每个控制步（1 秒），根据当前 noisy observation 做 **Bayesian belief update**。
3. 对所有候选动作计算 **Expected Free Energy（EFE）**——即这个动作执行后，预期观测和目标偏好之间的差距（pragmatic term），加上该动作能带来多少对隐藏状态的信息增益（epistemic term）。
4. 用 precision-weighted softmax 选动作。

这里的 EFE 是单步近似（T=1），不是完整的多步 policy inference——这是针对实时边缘控制的工程折中：Jetson 硬件上每步计算 <1ms，完全满足 1 秒控制步的实时要求。belief 本身携带了历史信息，所以控制器并非完全无记忆。

关键不是公式本身，而是生成模型里怎么定义"某个动作会带来什么系统级后果"。

### 5. 为什么有意义

实验上看到的现象很有代表性：

- 默认参数的 AIF（86.0% SLO）和 Heuristic（87.8%）同一量级。
- DQN 在 steady overload 场景平均可以到 94.9%，但 burst 场景跨 seed 方差极大（64.6%–94.3%），switch count 从 2 到 87 都有——这说明短期训练产生的策略本质上不可预测。
- **一旦把 OFFLOAD 和 SKIP 的 likelihood 从局部语义修正为系统级语义，AIF 的平均 SLO 从 86% 提到 93.2%，超过了带 profiling 信息的 myopic baseline（89.9%）。**

更重要的结论是：对这类边缘调度问题，**生成模型的动作语义编码本身就是方法的一部分**，不是无关紧要的实现细节。模型写对了什么，比选哪个算法更重要——两个 likelihood 矩阵的修正带来了约 16 pp 的提升，而所有算法参数（precision、epistemic weight、cooldown）合在一起贡献不到 8 pp。

---

## 一句话核心

> 这篇工作想回答的不是"AIF 能不能套进 edge scheduling"，而是：**当环境部分可观测时，隐藏状态推断和动作语义建模会不会真正改变调度行为。**
