# MAD 论文深度剖析：实验设置、写作逻辑与动机拆解

这份笔记对之前的四篇论文进行了极高颗粒度的拆解。重点关注**实验细节**（数据集、模型、框架配置）以及**背后的研究逻辑**（Motivation Work、为什么这样设计实验）。这对于你复现实验和构思新论文至关重要。

---

## 1. Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs
*   **文件名**: `2311.17371v3.pdf`

### 1.1 写作逻辑与动机 (Motivation & Logic)
*   **起始动机 (Motivation Work)**:
    *   作者注意到 MAD 很火，但以前的工作大多只和“Zero-shot”对比，或者只在单一领域（如数学）测试。
    *   **核心疑问**: 这种复杂的系统真的比简单的 Prompt 工程（如 Chain-of-Thought）或集成（Self-Consistency）更划算吗？
*   **推导逻辑**:
    1.  **基准测试**: 先把所有主流 MAD 方法拉出来，和最强的单 Agent 策略（Medprompt, SC）比一比。
    2.  **发现异常**: 发现 MAD 并不总是赢，而且成本巨高。
    3.  **归因分析 (Attribution)**: 既然架构没赢，是不是“人设”的问题？
    4.  **深入探究 (Deep Dive)**: 提出了 **"Agreement Modulation" (一致性调节)** 实验。这是本文的高光时刻。作者假设：Agent 之间太容易达成一致（盲从）或者太难达成一致（吵架），是影响性能的关键。
    5.  **最终结论**: 只要通过 Prompt 控制 Agent 的“顺从度”，原本最差的 Multi-Persona 就能变成最好的。

### 1.2 实验设置细节
*   **LLM 模型**:
    *   主力: `gpt-3.5-turbo` (为了省钱做大规模测试)
    *   验证: `gpt-4`, `Mixtral-8x7B` (证明结论可迁移)
*   **数据集 (7个)**:
    *   *医学类*: MedQA (USMLE), PubMedQA, MMLU-Clinical (专业知识强)
    *   *推理类*: CosmosQA (常识阅读理解), CIAR (反直觉算术), GPQA (专家级科学问题), Chess (棋局状态追踪)
*   **MAD 框架与配置**:
    *   **Society of Minds (SoM)**: 多个 Agent 独立回答 -> 汇总 -> 再次回答。
    *   **Multi-Persona**: 两个 Agent（天使 vs 魔鬼）+ 一个 Judge。
    *   **ChatEval**: 包括 One-on-one, Simultaneous-talk 等变体。
    *   **超参数**:
        *   Agent 数量: 2 到 4 个。
        *   轮数 (Rounds): 2 到 4 轮。
        *   Temperature: 未明确强调，通常设为 1.0 以增加多样性。

---

## 2. If Multi-Agent Debate is the Answer, What is the Question?
*   **文件名**: `2502.08788v2.pdf`

### 2.1 写作逻辑与动机 (Motivation & Logic)
*   **起始动机 (Motivation Work)**:
    *   现有的 MAD 论文都在“刷榜”，但使用的 Baseline 很弱（通常只比 Standard Prompting）。
    *   **批判性思维**: 如果我给单 Agent 同样的算力（Inference Budget），比如让它多采样几次（Self-Consistency），MAD 还能赢吗？
*   **推导逻辑**:
    1.  **复现与打假**: 严格控制 Token 消耗量，对比 MAD 和 Self-Consistency (SC)。结果发现：大多数 MAD 方法打不过 SC。
    2.  **寻找原因**: 为什么输了？分析 "Corrected Questions" (纠正的题) vs "Erred Questions" (改错的题)。发现 MAD 经常把做对的题改错了（引入了噪音）。
    3.  **提出假设**: 同质化（Homogeneous）的 Agent 知识库是一样的，辩论只是在“内部反刍”。
    4.  **验证假设 (Heter-MAD)**: 既然同质化不行，那就引入异构性。让 Agent 在每一步随机调用不同的模型（如 Llama 3 和 GPT-4 混用）。
    5.  **结论**: 异构性才是 MAD 真正的增长点，而不是辩论机制本身。

### 2.2 实验设置细节
*   **LLM 模型**:
    *   `gpt-4o-mini`, `claude-3.5-haiku`, `Llama-3.1-8b-instruct`, `Llama-3.1-70b-instruct` (覆盖闭源与开源、大与小)。
*   **数据集 (9个)**:
    *   *通用知识*: MMLU, MMLU-Pro, CommensenseQA, ARC-Challenge, AGIEval
    *   *数学*: GSM8K, MATH
    *   *代码*: HumanEval, MBPP
*   **MAD 框架与配置**:
    *   **框架**: SoM, Multi-Persona, Exchange-of-Thoughts (EoT), ChatEval, AgentVerse。
    *   **关键设置**:
        *   **算力对齐**: 强制 MAD 的 API 调用次数 = 6 次。对比的 Self-Consistency 也采样 6 次。这在逻辑上无懈可击。
        *   **Temperature**: `T=1.0`, `top-p=1.0` (为了保证多样性)。
    *   **Heter-MAD 设置**: Agent 在生成回答时，以 0.5 的概率随机选择 Model A 或 Model B。

---

## 3. Debate or Vote: Which Yields Better Decisions?
*   **文件名**: `2508.17536v2.pdf`

### 3.1 写作逻辑与动机 (Motivation & Logic)
*   **起始动机 (Motivation Work)**:
    *   MAD 包含两个部分：1. 多 Agent 生成 (Ensembling/Voting)，2. 相互交流 (Communication/Debate)。
    *   **核心解耦**: 性能提升到底来自于 1 还是 2？之前的论文把两者混为一谈。
*   **推导逻辑**:
    1.  **定义 Baseline**: 定义 "Majority Voting" 为 `Round=0` 的 MAD。
    2.  **实验对比**: 跑 `Round=0` (Voting) vs `Round=T` (Debate)。
    3.  **发现现象**: 绝大多数提升在 `Round=0` 就已经完成了，后续的 Debate 提升微乎其微，甚至下降。
    4.  **理论建模 (Theoretical Grounding)**: 为什么 Debate 没用？作者引入了 **鞅 (Martingale)** 理论。证明在贝叶斯更新下，Agent 对正确答案信念的期望值是不变的。这是一篇有很强理论深度的论文。
    5.  **改进设计**: 既然理论上是鞅（无效），那必须引入“偏置（Bias）”。设计 `MAD-Conformist`（从众策略），强制让 Agent 向大多数人的意见靠拢，人为打破鞅的平衡。

### 3.2 实验设置细节
*   **LLM 模型**:
    *   `Qwen2.5-7B-Instruct`, `Llama-3.1-8B-Instruct`, `Qwen2.5-32B-Instruct` (侧重开源模型)。
*   **数据集 (7个)**:
    *   Arithmetics, GSM8K, MMLU (Pro-Med, Formal-Logic), HellaSwag, CommonSenseQA, HH-RLHF (对齐任务).
*   **MAD 框架与配置**:
    *   **框架**: Decentralized MAD (全连接), Sparse MAD (环状拓扑), Centralized MAD (星型拓扑)。
    *   **变量**: Agent 数量 (N=5), 轮数 (T=2, 3, 5)。
    *   **关键操作**: 使用特定的 Prompt 提取答案格式 `{final answer: ...}`，避免因为解析错误导致的评估偏差。

---

## 4. Talk Isn’t Always Cheap: Understanding Failure Modes
*   **文件名**: `2509.05396v1.pdf`

### 4.1 写作逻辑与动机 (Motivation & Logic)
*   **起始动机 (Motivation Work)**:
    *   大家通常认为“真理越辩越明”。但是在 LLM 里，强模型会不会被弱模型带偏？
    *   **反向思考**: 关注 MAD 的“失效模式”（Failure Modes），而不是成功模式。
*   **推导逻辑**:
    1.  **构建异构环境**: 强模型 (GPT-4) + 弱模型 (Mistral)。
    2.  **观察动态**: 随着辩论进行，准确率是上升还是下降？
    3.  **微观分析**: 追踪每一道题的状态流转。
        *   `Correct -> Incorrect` (被带偏)：红色警报。
        *   `Incorrect -> Correct` (被纠正)：有效辩论。
    4.  **发现 Sycophancy (阿谀奉承)**: 发现强模型经常为了“合群”而放弃自己的正确答案。这可能源于 RLHF 训练让模型变得过于顺从。

### 4.2 实验设置细节
*   **LLM 模型**:
    *   `GPT-4o-mini`, `Llama-3.1-8B-Instruct`, `Mistral-7B-v0.2`.
*   **数据集**:
    *   CommonSenseQA (CSQA), MMLU, GSM8K.
*   **MAD 框架与配置**:
    *   **组合方式**:
        *   3 Mistral (全弱)
        *   3 GPT-4 (全强)
        *   1 GPT + 2 Llama (强被弱包围)
        *   2 GPT + 1 Llama (强包围弱)
    *   **Prompt**: 明确要求 Agent "Using the reasoning from other agents as additional advice" (使用他人的推理作为额外建议)。

---

## 5. 总结：给你的论文写作建议

基于这四篇论文，如果你想做 MAD 相关的研究，以下是**必须**考虑的实验逻辑：

1.  **Baseline 必须硬**:
    *   不要只对比 "Zero-shot"。
    *   **必须对比 Self-Consistency (SC)**。如果你的 MAD 跑了 N 个 Agent * T 轮，那么 SC 也必须采样 N*T 次。打不过 SC，你的方法就没有实用价值。
    *   **必须对比 Majority Voting**: 证明你的提升是来自“交互”，而不是“人多”。

2.  **异构性是趋势 (Heterogeneity)**:
    *   全用 GPT-4 的实验已经过时了。现在的趋势是“大小模型搭配”或者“不同家族模型搭配”。
    *   研究“强带弱”或“弱带强”的动力学。

3.  **深入机理分析**:
    *   不要只放准确率表格。
    *   **做 Transition Matrix**: 画出 `Correct->Incorrect` 和 `Incorrect->Correct` 的比例图。
    *   **做 Token/Cost 分析**: 算一下提升 1% 的准确率需要多花多少钱。

4.  **理论或失效分析**:
    *   像论文 3 那样尝试用数学解释（鞅过程）。
    *   像论文 4 那样分析社会学现象（盲从、回声室效应）。

这些细节都在 Markdown 文件中详细列出了，你可以作为 Checklist 来检查你的实验设计。