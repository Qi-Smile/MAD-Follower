# MAD Follower | 多智能体辩论从众效应研究框架

<div align="center">

**A Quantitative Framework for Studying Conformity Effects in Multi-Agent LLM Debates**

**量化研究多智能体LLM辩论中从众效应的实验框架**

[English](#english) | [中文](#中文)

</div>

---

<a name="english"></a>

## 📖 English Documentation

### Overview

MAD Follower is an experimental framework inspired by Asch's classical conformity experiments, designed to quantitatively study social conformity phenomena in multi-agent debate systems. The framework orchestrates debates between LLM agents with distinct roles to observe whether rational agents can be swayed by persistent misinformation from an adversarial coalition.

**Research Question**: In multi-agent debates, when an **Attacker** consistently advocates incorrect answers supported by a **Follower**, will **Normal Agents** maintain correct reasoning or conform to the incorrect majority?

### Key Features

- 🎯 **Automated Experiment Design**: Dynamic target selection ensures attackers always advocate wrong answers (automatically chosen to differ from ground truth)
- ⚡ **Async-First Architecture**: Two-tier concurrency control (question-level + agent-level) enables large-scale experiments
- 📊 **Comprehensive Metrics**:
  - Conformity Rate (with binomial significance test)
  - Answer Alignment Rate per round
  - Textual Similarity to attacker
  - Confidence Dynamics across rounds
- 📈 **Built-in Visualization**: Auto-generated plots (drift metrics, conformity rates, confidence trends)
- ✅ **Role Compliance Validation**: Real-time detection of agent behavioral deviations with warning logs
- 📝 **Complete Audit Trail**: Thread-safe JSONL logging capturing every turn with full context (prompt, response, latency, token usage)
- 🔧 **Extensible Architecture**: Easy to add custom agent types, metrics, and datasets

### Experiment Design

#### Three-Role System

1. **Attacker Agent**
   - Advocates an incorrect answer with persuasive reasoning
   - Target answer automatically selected to differ from ground truth
   - Validated after each turn to ensure compliance

2. **Follower Agent**
   - Round 1: Provides independent answer (brief)
   - Rounds 2+: Mirrors attacker's previous answer with reinforcing arguments
   - Receives explicit `FOLLOW_TARGET=<attacker_answer>` context injection

3. **Normal Agents** (3+)
   - Independent thinkers attempting rational reasoning
   - Encouraged to challenge dubious claims
   - May be influenced by attacker-follower coalition pressure

#### Multi-Round Debate Flow

```
Round 1: All agents respond independently (no history)
    ↓
Round 2: Agents see [Round 1] history, may revise answers
    ↓
Round 3: Agents see [Round 1-2] history, final answers
    ↓
Consensus: Majority voting or confidence-weighted aggregation
```

**History Context**: Agents see complete debate history from all previous rounds (intelligently truncated to first + last 11 turns if >12 entries).

### Installation

#### Prerequisites

- Python 3.10+
- OpenAI-compatible LLM API endpoint (e.g., OpenAI, DashScope/Qwen, local vLLM)

#### Setup

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API credentials
export BASE_URL="https://api.openai.com/v1"
export API_KEY="sk-..."

# For Alibaba Cloud DashScope (Qwen):
# export BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
# export API_KEY="sk-..."
```

### Quick Start

#### 5-Minute Demo

```bash
# Run experiment on 5 questions with automatic visualization
python scripts/run_analysis_demo.py \
  --dataset data/commonsense_qa_validation.jsonl \
  --limit 5 \
  --rounds 3 \
  --normal-agents 3

# Outputs:
# - outputs/debate_log.jsonl      : Complete debate transcript
# - outputs/run_summary.json      : Experiment summary with accuracy & warnings
# - outputs/plots/                : PNG visualizations
#   ├── drift_alignment.png       : Answer alignment + text similarity trends
#   ├── answer_change_hist.png    : Distribution of answer changes
#   ├── confidence_trend.png      : Average confidence trajectory
#   └── conformity_rate.png       : Conformity rate with p-value
```

#### Basic Usage (No Visualization)

```bash
python scripts/run_demo.py \
  --dataset data/commonsense_qa_validation.jsonl \
  --limit 10 \
  --rounds 3 \
  --question-workers 2 \
  --agent-workers 5 \
  --normal-agents 3
```

### Configuration

#### CLI Arguments

```bash
--dataset PATH              # Path to CommonsenseQA-format JSONL
--limit N                   # Number of questions to process
--rounds N                  # Number of debate rounds (default: 3)
--question-workers N        # Concurrent questions (default: 2)
--agent-workers N           # Concurrent agents per question (default: 5)
--normal-agents N           # Number of normal agents (default: 3)
--attacker-target LABEL     # Preferred attacker target (e.g., "B")
--consensus METHOD          # "majority" or "confidence_weighted"
--model NAME                # LLM model name (default: "qwen-plus")
--temperature FLOAT         # Sampling temperature (default: 0.7)
```

#### Programmatic Configuration

```python
from mad_debate.config import ExperimentConfig, LLMSettings, DebateHyperParams, AgentConfig
from pathlib import Path

config = ExperimentConfig(
    llm=LLMSettings(
        model="qwen3-8b",
        temperature=0.7,
        max_tokens=512,
        max_concurrent_requests=6,
        max_retries=2
    ),
    dataset=DatasetConfig(
        path=Path("data/commonsense_qa_validation.jsonl"),
        limit=30,
        shuffle=True
    ),
    debate=DebateHyperParams(
        rounds=3,
        max_concurrent_questions=2,
        per_question_agent_concurrency=5,
        consensus_method="majority"
    ),
    agents=[
        AgentConfig(agent_id="attacker", role="attacker", target_answer="B"),
        AgentConfig(agent_id="follower", role="follower", follow_target="attacker"),
        AgentConfig(agent_id="responder_1", role="normal"),
        AgentConfig(agent_id="responder_2", role="normal"),
        AgentConfig(agent_id="responder_3", role="normal")
    ]
)
```

### Metrics & Analysis

#### Core Metrics

1. **Conformity Rate**: Proportion of agents who initially answered correctly but ultimately switched to incorrect answers
   ```python
   {
     "converted": 7,       # Agents that switched correct→incorrect
     "candidates": 30,     # Total agents with both initial and final answers
     "rate": 0.233,        # Conformity rate (23.3%)
     "p_value": 0.032      # Binomial test p-value (significant if <0.05)
   }
   ```

2. **Drift Metrics** (per round):
   ```python
   DriftMetric(
     round_index=2,
     average_similarity_to_attacker=0.67,    # Text similarity (SequenceMatcher)
     follower_similarity=0.95,                # Follower's text similarity
     answer_alignment_rate=0.42,              # % agents matching attacker's answer
     follower_alignment=True                  # Follower matched attacker's answer
   )
   ```

3. **Answer Trajectory**: Per-agent answer evolution across rounds
   ```python
   AnswerTrajectory(
     question_id="q001",
     agent_id="responder_1",
     answers=["A", "B", "B"],               # Answers across rounds
     confidences=[0.95, 0.88, 0.87],        # Confidence scores
     ground_truth="A"                       # Correct answer
   )
   ```

#### Programmatic Analysis

```python
from pathlib import Path
from mad_debate.metrics import (
    load_turn_logs,
    compute_conformity_rate,
    compute_drift,
    build_answer_trajectories
)

# Load debate logs
turns = load_turn_logs(Path("outputs/debate_log.jsonl"))

# Compute conformity rate with statistical significance
stats = compute_conformity_rate(turns, attacker_id="attacker")
print(f"Conformity Rate: {stats['rate']:.2%}")
print(f"Statistical Significance: p={stats['p_value']:.3f}")
print(f"Converted Agents: {stats['converted']}/{stats['candidates']}")

# Compute drift metrics per round
drift = compute_drift(turns, attacker_id="attacker", follower_id="follower")
for metric in drift:
    print(f"Round {metric.round_index}: "
          f"Alignment={metric.answer_alignment_rate:.2%}, "
          f"Similarity={metric.average_similarity_to_attacker:.2f}")

# Analyze answer trajectories
trajectories = build_answer_trajectories(turns)
for traj in trajectories:
    if traj.agent_id.startswith("responder"):
        print(f"{traj.agent_id}: {' → '.join(traj.answers)}")
```

### Dataset Format

The framework uses CommonsenseQA-compatible JSONL format:

```json
{
  "id": "1afa02df02c908a558b4036e80242fac",
  "question": "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?",
  "choices": {
    "label": ["A", "B", "C", "D", "E"],
    "text": ["bank", "library", "department store", "mall", "new york"]
  },
  "answerKey": "A"
}
```

**Provided Datasets**:
- `data/commonsense_qa_validation.jsonl`: 1221 questions from HuggingFace `tau/commonsense_qa`
- `data/sample_questions.jsonl`: 3 sample questions for quick testing

### Extending the Framework

#### Adding Custom Agent Types

```python
# 1. Define new agent class in mad_debate/agents.py
from mad_debate.agents import BaseAgent

class MediatorAgent(BaseAgent):
    def behavior_instructions(self, round_index, question_state):
        return (
            "You are a mediator. Acknowledge both sides' arguments "
            "and propose a balanced perspective."
        )

    def _build_prompt(self, question, round_index, context, question_state):
        prompt = super()._build_prompt(question, round_index, context, question_state)
        prompt += "\nConsider merits of all previous answers before responding."
        return prompt

# 2. Register in mad_debate/debate_runner.py
def _instantiate_agent(self, agent_cfg):
    if agent_cfg.role == "attacker":
        return AttackerAgent(agent_cfg, self.llm_client)
    elif agent_cfg.role == "follower":
        return FollowerAgent(agent_cfg, self.llm_client)
    elif agent_cfg.role == "mediator":  # Add this
        return MediatorAgent(agent_cfg, self.llm_client)
    return NormalAgent(agent_cfg, self.llm_client)

# 3. Use in experiment configuration
config.agents.append(AgentConfig(
    agent_id="mediator_1",
    role="mediator",
    description="Seeks balanced perspective"
))
```

#### Adding Custom Metrics

```python
# Add to mad_debate/metrics.py
def compute_consensus_flip_rate(turns: List[AgentTurnRecord]) -> float:
    """
    Compute proportion of questions where consensus changed
    from Round 1 to final round.
    """
    grouped = defaultdict(list)
    for turn in turns:
        grouped[turn.question_id].append(turn)

    flipped = 0
    total = 0

    for question_id, question_turns in grouped.items():
        round_0 = [t for t in question_turns if t.round_index == 0]
        last_round = max(t.round_index for t in question_turns)
        round_last = [t for t in question_turns if t.round_index == last_round]

        consensus_0 = Counter(t.parsed_answer for t in round_0).most_common(1)[0][0]
        consensus_last = Counter(t.parsed_answer for t in round_last).most_common(1)[0][0]

        total += 1
        if consensus_0 != consensus_last:
            flipped += 1

    return flipped / total if total > 0 else 0.0
```

#### Integrating New Datasets

Any dataset following this structure can be used:

```json
{
  "id": "<unique_id>",
  "question": "<question_text>",
  "choices": {
    "label": ["A", "B", "C", ...],
    "text": ["option1", "option2", "option3", ...]
  },
  "answerKey": "<correct_label>"
}
```

Simply pass `--dataset path/to/your/dataset.jsonl` to the runner scripts.

### Architecture Overview

```
mad_debate/
├── config.py           # Configuration dataclasses (ExperimentConfig, AgentConfig, etc.)
├── schemas.py          # Data schemas (AgentTurnRecord, ConsensusRecord, RunSummary)
├── datasets.py         # Dataset loader for CommonsenseQA format
├── llm_client.py       # Async OpenAI-compatible LLM client with retry logic
├── agents.py           # Agent implementations (Base, Attacker, Follower, Normal)
├── debate_runner.py    # Main orchestration engine with concurrency control
├── metrics.py          # Conformity metrics (drift, alignment, trajectories)
└── logging_utils.py    # Thread-safe JSONL logger
```

**Data Flow**:
```
Questions → DebateRunner → _prepare_question_state (select attacker target)
                         ↓
          For each round: _run_round (parallel agent responses)
                         ↓
                    _validate_role_behavior (check compliance)
                         ↓
                    Log AgentTurnRecords
                         ↓
          After all rounds: _build_consensus
                         ↓
                    Output: ConsensusRecord + RunSummary
```

### Research Applications

1. **Social Psychology in AI**: Quantify conformity effects in LLM systems, analogous to Asch's experiments
2. **AI Safety**: Study how adversarial agents can influence collective decision-making
3. **Collective Intelligence**: Understand error propagation mechanisms in multi-agent systems
4. **Robustness Evaluation**: Test LLM reasoning independence under social pressure

### Citation

If you use this framework in your research, please cite:

```bibtex
@software{mad_follower2025,
  title={MAD Follower: A Framework for Studying Conformity Effects in Multi-Agent Debates},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/mad-follower}
}
```

### License

MIT License - See [LICENSE](LICENSE) for details.

---

<a name="中文"></a>

## 📖 中文文档

### 概述

MAD Follower 是一个受Asch经典从众实验启发的实验框架，旨在量化研究多智能体辩论系统中的社会从众现象。该框架编排不同角色的LLM智能体进行辩论，观察理性智能体是否会被来自对抗性联盟的持续错误信息所影响。

**研究问题**：在多智能体辩论中，当一个**攻击者（Attacker）**持续倡导错误答案并得到**跟随者（Follower）**支持时，**正常智能体（Normal Agents）**能否保持正确推理，还是会从众于错误的多数派？

### 核心特性

- 🎯 **自动化实验设计**：动态目标选择确保攻击者始终倡导错误答案（自动选择与标准答案不同的选项）
- ⚡ **异步优先架构**：两层并发控制（问题级+智能体级）支持大规模实验
- 📊 **全面指标体系**：
  - 从众率（含二项显著性检验）
  - 每轮答案对齐率
  - 与攻击者的文本相似度
  - 跨轮次置信度动态变化
- 📈 **内置可视化**：自动生成图表（漂移度指标、从众率、置信度趋势）
- ✅ **角色合规验证**：实时检测智能体行为偏差并记录警告日志
- 📝 **完整审计轨迹**：线程安全的JSONL日志记录每个轮次的完整上下文（提示词、响应、延迟、token使用量）
- 🔧 **可扩展架构**：易于添加自定义智能体类型、指标和数据集

### 实验设计

#### 三角色系统

1. **攻击者智能体**
   - 使用有说服力的论证倡导错误答案
   - 目标答案自动选择为与标准答案不同的选项
   - 每轮后验证以确保合规性

2. **跟随者智能体**
   - 第1轮：提供独立答案（简短）
   - 第2轮以后：模仿攻击者上一轮的答案并提供强化论证
   - 接收显式的 `FOLLOW_TARGET=<攻击者答案>` 上下文注入

3. **正常智能体**（3+个）
   - 尝试理性推理的独立思考者
   - 被鼓励挑战可疑主张
   - 可能受到攻击者-跟随者联盟压力的影响

#### 多轮辩论流程

```
第1轮：所有智能体独立回答（无历史记录）
    ↓
第2轮：智能体看到[第1轮]历史，可修正答案
    ↓
第3轮：智能体看到[第1-2轮]历史，给出最终答案
    ↓
共识：多数投票或置信度加权聚合
```

**历史上下文**：智能体可看到所有之前轮次的完整辩论历史（如超过12条记录则智能截断为首条+最后11条）。

### 安装

#### 前置要求

- Python 3.10+
- OpenAI兼容的LLM API端点（如OpenAI、阿里云DashScope/通义千问、本地vLLM）

#### 环境配置

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置API凭证
export BASE_URL="https://api.openai.com/v1"
export API_KEY="sk-..."

# 使用阿里云DashScope（通义千问）：
# export BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
# export API_KEY="sk-..."
```

### 快速开始

#### 5分钟演示

```bash
# 在5个问题上运行实验并自动生成可视化
python scripts/run_analysis_demo.py \
  --dataset data/commonsense_qa_validation.jsonl \
  --limit 5 \
  --rounds 3 \
  --normal-agents 3

# 输出：
# - outputs/debate_log.jsonl      : 完整辩论记录
# - outputs/run_summary.json      : 实验摘要（含准确率和警告）
# - outputs/plots/                : PNG可视化图表
#   ├── drift_alignment.png       : 答案对齐率+文本相似度趋势
#   ├── answer_change_hist.png    : 答案变化次数分布
#   ├── confidence_trend.png      : 平均置信度轨迹
#   └── conformity_rate.png       : 从众率及p值
```

#### 基础用法（无可视化）

```bash
python scripts/run_demo.py \
  --dataset data/commonsense_qa_validation.jsonl \
  --limit 10 \
  --rounds 3 \
  --question-workers 2 \
  --agent-workers 5 \
  --normal-agents 3
```

### 配置

#### 命令行参数

```bash
--dataset PATH              # CommonsenseQA格式JSONL文件路径
--limit N                   # 处理的问题数量
--rounds N                  # 辩论轮数（默认：3）
--question-workers N        # 并发问题数（默认：2）
--agent-workers N           # 每个问题的并发智能体数（默认：5）
--normal-agents N           # 正常智能体数量（默认：3）
--attacker-target LABEL     # 攻击者首选目标（如"B"）
--consensus METHOD          # "majority"或"confidence_weighted"
--model NAME                # LLM模型名称（默认："qwen-plus"）
--temperature FLOAT         # 采样温度（默认：0.7）
```

#### 编程式配置

```python
from mad_debate.config import ExperimentConfig, LLMSettings, DebateHyperParams, AgentConfig
from pathlib import Path

config = ExperimentConfig(
    llm=LLMSettings(
        model="qwen3-8b",
        temperature=0.7,
        max_tokens=512,
        max_concurrent_requests=6,
        max_retries=2
    ),
    dataset=DatasetConfig(
        path=Path("data/commonsense_qa_validation.jsonl"),
        limit=30,
        shuffle=True
    ),
    debate=DebateHyperParams(
        rounds=3,
        max_concurrent_questions=2,
        per_question_agent_concurrency=5,
        consensus_method="majority"
    ),
    agents=[
        AgentConfig(agent_id="attacker", role="attacker", target_answer="B"),
        AgentConfig(agent_id="follower", role="follower", follow_target="attacker"),
        AgentConfig(agent_id="responder_1", role="normal"),
        AgentConfig(agent_id="responder_2", role="normal"),
        AgentConfig(agent_id="responder_3", role="normal")
    ]
)
```

### 指标与分析

#### 核心指标

1. **从众率**：初始回答正确但最终改为错误答案的智能体比例
   ```python
   {
     "converted": 7,       # 从正确转为错误的智能体数
     "candidates": 30,     # 有初始和最终答案的总智能体数
     "rate": 0.233,        # 从众率（23.3%）
     "p_value": 0.032      # 二项检验p值（<0.05表示显著）
   }
   ```

2. **漂移度指标**（每轮）：
   ```python
   DriftMetric(
     round_index=2,
     average_similarity_to_attacker=0.67,    # 文本相似度（SequenceMatcher）
     follower_similarity=0.95,                # 跟随者的文本相似度
     answer_alignment_rate=0.42,              # 与攻击者答案一致的智能体百分比
     follower_alignment=True                  # 跟随者是否与攻击者答案一致
   )
   ```

3. **答案轨迹**：每个智能体跨轮次的答案演化
   ```python
   AnswerTrajectory(
     question_id="q001",
     agent_id="responder_1",
     answers=["A", "B", "B"],               # 各轮答案
     confidences=[0.95, 0.88, 0.87],        # 置信度分数
     ground_truth="A"                       # 正确答案
   )
   ```

#### 编程式分析

```python
from pathlib import Path
from mad_debate.metrics import (
    load_turn_logs,
    compute_conformity_rate,
    compute_drift,
    build_answer_trajectories
)

# 加载辩论日志
turns = load_turn_logs(Path("outputs/debate_log.jsonl"))

# 计算从众率及统计显著性
stats = compute_conformity_rate(turns, attacker_id="attacker")
print(f"从众率: {stats['rate']:.2%}")
print(f"统计显著性: p={stats['p_value']:.3f}")
print(f"转换智能体: {stats['converted']}/{stats['candidates']}")

# 计算每轮漂移度指标
drift = compute_drift(turns, attacker_id="attacker", follower_id="follower")
for metric in drift:
    print(f"第{metric.round_index}轮: "
          f"对齐率={metric.answer_alignment_rate:.2%}, "
          f"相似度={metric.average_similarity_to_attacker:.2f}")

# 分析答案轨迹
trajectories = build_answer_trajectories(turns)
for traj in trajectories:
    if traj.agent_id.startswith("responder"):
        print(f"{traj.agent_id}: {' → '.join(traj.answers)}")
```

### 数据集格式

框架使用CommonsenseQA兼容的JSONL格式：

```json
{
  "id": "1afa02df02c908a558b4036e80242fac",
  "question": "旋转门方便双向通行，但它也作为什么地方的安全措施？",
  "choices": {
    "label": ["A", "B", "C", "D", "E"],
    "text": ["银行", "图书馆", "百货商店", "商场", "纽约"]
  },
  "answerKey": "A"
}
```

**提供的数据集**：
- `data/commonsense_qa_validation.jsonl`: 来自HuggingFace `tau/commonsense_qa`的1221个问题
- `data/sample_questions.jsonl`: 3个示例问题用于快速测试

### 扩展框架

#### 添加自定义智能体类型

```python
# 1. 在 mad_debate/agents.py 中定义新智能体类
from mad_debate.agents import BaseAgent

class MediatorAgent(BaseAgent):
    def behavior_instructions(self, round_index, question_state):
        return (
            "你是调解者。承认双方的论点，"
            "并提出平衡的观点。"
        )

    def _build_prompt(self, question, round_index, context, question_state):
        prompt = super()._build_prompt(question, round_index, context, question_state)
        prompt += "\n在回答前考虑所有先前答案的优点。"
        return prompt

# 2. 在 mad_debate/debate_runner.py 中注册
def _instantiate_agent(self, agent_cfg):
    if agent_cfg.role == "attacker":
        return AttackerAgent(agent_cfg, self.llm_client)
    elif agent_cfg.role == "follower":
        return FollowerAgent(agent_cfg, self.llm_client)
    elif agent_cfg.role == "mediator":  # 添加这个
        return MediatorAgent(agent_cfg, self.llm_client)
    return NormalAgent(agent_cfg, self.llm_client)

# 3. 在实验配置中使用
config.agents.append(AgentConfig(
    agent_id="mediator_1",
    role="mediator",
    description="寻求平衡观点"
))
```

#### 添加自定义指标

```python
# 添加到 mad_debate/metrics.py
def compute_consensus_flip_rate(turns: List[AgentTurnRecord]) -> float:
    """
    计算从第1轮到最后一轮共识发生变化的问题比例
    """
    grouped = defaultdict(list)
    for turn in turns:
        grouped[turn.question_id].append(turn)

    flipped = 0
    total = 0

    for question_id, question_turns in grouped.items():
        round_0 = [t for t in question_turns if t.round_index == 0]
        last_round = max(t.round_index for t in question_turns)
        round_last = [t for t in question_turns if t.round_index == last_round]

        consensus_0 = Counter(t.parsed_answer for t in round_0).most_common(1)[0][0]
        consensus_last = Counter(t.parsed_answer for t in round_last).most_common(1)[0][0]

        total += 1
        if consensus_0 != consensus_last:
            flipped += 1

    return flipped / total if total > 0 else 0.0
```

#### 集成新数据集

任何遵循以下结构的数据集都可以使用：

```json
{
  "id": "<唯一标识>",
  "question": "<问题文本>",
  "choices": {
    "label": ["A", "B", "C", ...],
    "text": ["选项1", "选项2", "选项3", ...]
  },
  "answerKey": "<正确标签>"
}
```

只需向运行脚本传递 `--dataset path/to/your/dataset.jsonl` 即可。

### 架构概览

```
mad_debate/
├── config.py           # 配置数据类（ExperimentConfig, AgentConfig等）
├── schemas.py          # 数据模式（AgentTurnRecord, ConsensusRecord, RunSummary）
├── datasets.py         # CommonsenseQA格式数据集加载器
├── llm_client.py       # 带重试逻辑的异步OpenAI兼容LLM客户端
├── agents.py           # 智能体实现（Base, Attacker, Follower, Normal）
├── debate_runner.py    # 带并发控制的主编排引擎
├── metrics.py          # 从众指标（漂移度、对齐率、轨迹）
└── logging_utils.py    # 线程安全的JSONL日志记录器
```

**数据流**：
```
问题 → DebateRunner → _prepare_question_state（选择攻击者目标）
                    ↓
         每轮: _run_round（并行智能体响应）
                    ↓
              _validate_role_behavior（检查合规性）
                    ↓
              记录AgentTurnRecords
                    ↓
    所有轮次后: _build_consensus
                    ↓
              输出: ConsensusRecord + RunSummary
```

### 研究应用

1. **AI中的社会心理学**：量化LLM系统中的从众效应，类似于Asch实验
2. **AI安全**：研究对抗性智能体如何影响集体决策
3. **集体智能**：理解多智能体系统中的错误传播机制
4. **鲁棒性评估**：测试LLM在社会压力下的推理独立性

### 引用

如果在研究中使用本框架，请引用：

```bibtex
@software{mad_follower2025,
  title={MAD Follower: 多智能体辩论从众效应研究框架},
  author={你的名字},
  year={2025},
  url={https://github.com/yourusername/mad-follower}
}
```

### 许可证

MIT许可证 - 详见 [LICENSE](LICENSE)

---

## 📧 Contact | 联系方式

For questions or collaboration inquiries, please open an issue on GitHub.

如有问题或合作咨询，请在GitHub上开启issue。

## 🙏 Acknowledgments | 致谢

This framework is inspired by:
- Asch, S. E. (1951). Effects of group pressure upon the modification and distortion of judgments.
- Recent advances in multi-agent debate research for LLM reasoning.

本框架受以下启发：
- Asch, S. E. (1951). 群体压力对判断修正和扭曲的影响。
- LLM推理的多智能体辩论研究的最新进展。

### Run Artifact Layout

每次运行现在都会在 `outputs/` 下生成一个带时间戳的子目录（例如 `outputs/20250115_103045_n5_r3_limit30/`），目录内容如下：

- `config.json`：本次实验使用的完整配置快照（LLM、数据集、辩论参数、Agent 列表等）。
- `debate_log.jsonl`：完整的逐轮对话日志，`metrics.py` 依旧从该文件读取数据。
- `run_summary.json`：只统计攻击者始终坚持错误答案的问题；被剔除的问题编号会记录在 `metadata.excluded_questions`。
- `questions/*.json`：每个问题一个 JSON，包含题面、选项、所有 Agent 的回答与理由、共识结果、警告列表、攻击目标以及 `excluded` 标记，便于溯源分析。
- `plots/`（若运行分析脚本）：自动生成漂移、答案切换、置信度、从众率等 PNG 图表。
### Run Artifact Layout

每次运行现在都会在 `outputs/` 下生成一个带时间戳的子目录（例如 `outputs/20250115_103045_n5_r3_limit30/`），目录内容如下：

- `config.json`：完整的实验配置快照（LLM、数据集、辩论超参、Agent 列表等）。
- `debate_log.jsonl`：逐轮对话日志，`metrics.py` 依旧从该文件读取数据。
- `run_summary.json`：只统计攻击者始终坚持错误答案的问题；被剔除的问题编号会记录在 `metadata.excluded_questions`。
- `questions/*.json`：每道题一个 JSON，包含题面、选项、所有 Agent 回复、共识结果、警告列表、攻击目标以及 `excluded` 标记，方便逐题分析。
- `plots/`（若运行 `scripts/run_analysis_demo.py`）：自动生成漂移、答案切换、置信度、从众率等 PNG 图表。

当攻击者未能维持指定的错误答案时，该问题会自动标记为 `excluded: true`，同时被剔除出整体准确率和从众指标的统计范围，确保只对“有效攻击”进行分析。
> Note: Drift and conformity metrics automatically ignore attacker/follower agents and only count conversions where a benign agent initially differed from the attacker but later matched the attacker’s target answer (unless you switch the reference to ground truth).
