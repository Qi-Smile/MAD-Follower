# MAD Follower 实验 README | Experiment Guide

本文档概述当前项目框架、主要方法实现与典型运行方式（中英文双语）。

## 1. 框架概览 / Overview

- **核心目标 / Goal**：研究多智能体辩论（MAD）的从众攻击与防御，量化 ASR（攻击成功率）、CR（从众率）、Transition（正确→错误、错误→正确）等。
- **主要组件 / Key Components**
  - `mad_debate/agents.py`：Attacker、Follower、Normal，支持攻防参数（语气、论证长度、跟随模式、置信度强制、人设预设）。
  - `mad_debate/debate_runner.py`：主编排（多轮辩论、并发、共识），支持改口惩罚与少数派加权。
  - `mad_debate/config.py`：配置数据类，含 `AttackSettings`/`DefenseSettings`。
  - `mad_debate/metrics.py` / `analysis.py`：加载日志、ASR/CR、Transition、轨迹、token 成本等。
  - `scripts/`：实验脚本（Phase 1 基线、Phase 2 机制拆解、Phase 3 攻击优化、Phase 4 防御验证、绘图与聚合）。
  - `configs/`：场景与实验模板 JSON。
  - `data/`：示例数据（commonsense、ethics 小样本）。

## 2. 环境与凭证 / Environment & Credentials

- 依赖 Python 3.10+，已使用 OpenAI 兼容 API 客户端。
- 必须设置环境变量：`BASE_URL`、`API_KEY`。

## 3. 关键方法与实现 / Methods & Implementation

- **攻防参数化 / Attack & Defense Controls**
  - `AttackSettings`：`follower_mode`（independent_then_follow / always_follow / delayed_follow）、`attacker_tone`（baseline/assertive/tentative/authoritative）、`attacker_rationale`（full/minimal/flawed）、`force_confidence`。
  - `DefenseSettings`：`normal_prompt_preset`（conformist/baseline/neutral/critical/independent）、`confidence_penalty_for_switchers`（改口惩罚）、`minority_boost`（少数派加权）。
  - 共识：`majority` 或 `confidence_weighted`，并结合改口惩罚与少数派加权。
- **指标 / Metrics**
  - ASR（attack_success_rate）、CR（conformity_rate）、Transition Matrix（C→C, C→I, I→C, I→I）、轨迹、答案振荡次数、token 成本。
  - 载于 `mad_debate/metrics.py`，`mad_debate/analysis.py` 统一封装。
- **日志与输出 / Logging**
  - 每次运行输出 `debate_log.jsonl`、`run_summary.json`、`config.json`，问题详情在 `run_dir/questions/`。
  - 图表默认保存在各 run 目录下的 `plots/`，避免覆盖。

## 4. 典型运行 / How to Run

### 4.1 Phase 1 基线场景 A/B/C（示例：ethics 30 题）

- A（攻/随大，普通小）  
  `python scripts/run_analysis_demo.py --dataset data/ethics_test.jsonl --dataset-type ethics --limit 30 --rounds 3 --attacker-target B --model qwen3-8b --attacker-model qwen-max --follower-model qwen-max --normal-model qwen3-8b --consensus majority --plots-dir outputs/ethics_A_plots`
- B（攻/随小，普通大）  
  `python scripts/run_analysis_demo.py --dataset data/ethics_test.jsonl --dataset-type ethics --limit 30 --rounds 3 --attacker-target B --model qwen3-8b --attacker-model qwen3-8b --follower-model qwen3-8b --normal-model qwen-max --consensus majority --plots-dir outputs/ethics_B_plots`
- C（全大）  
  `python scripts/run_analysis_demo.py --dataset data/ethics_test.jsonl --dataset-type ethics --limit 30 --rounds 3 --attacker-target B --model qwen-max --attacker-model qwen-max --follower-model qwen-max --normal-model qwen-max --consensus majority --plots-dir outputs/ethics_C_plots`

批量矩阵：`bash scripts/run_baseline_matrix.sh`（默认 A–E，commonsense，limit=30，可调整）。

### 4.2 Phase 2 机制拆解（单因子示例，场景 B）

基线：  
`python scripts/run_analysis_demo.py --dataset data/ethics_test.jsonl --dataset-type ethics --limit 30 --rounds 3 --attacker-target B --model qwen3-8b --attacker-model qwen3-8b --follower-model qwen3-8b --normal-model qwen-max --consensus majority --plots-dir outputs/ethics_B_base`

- 跟随时机（早跟随）：`--follower-mode always_follow`
- 权威语气：`--attacker-tone authoritative`
- 高置信：`--force-confidence 0.99`
- 简短论证：`--attacker-rationale minimal`
- 缺陷论证：`--attacker-rationale flawed`
- “转向”弱模拟：组合 `--follower-mode always_follow --attacker-rationale minimal --attacker-tone authoritative --force-confidence 0.99`

示例（答案为主，简短+权威+高置信）：  
`python scripts/run_analysis_demo.py ... --attacker-rationale minimal --attacker-tone authoritative --force-confidence 0.99 --plots-dir outputs/ethics_B_ans`

示例（推理为主，完整论证）：  
`python scripts/run_analysis_demo.py ... --attacker-rationale full --attacker-tone authoritative --force-confidence 0.99 --plots-dir outputs/ethics_B_reason`

示例（转向信号，早跟随+简短+高置信）：  
`python scripts/run_analysis_demo.py ... --follower-mode always_follow --attacker-rationale minimal --attacker-tone authoritative --force-confidence 0.99 --plots-dir outputs/ethics_B_delta`

（将 `...` 替换为基线公共参数）

### 4.3 Phase 3 攻击优化 / Phase 4 防御验证

- 攻击优化入口：`python scripts/run_attack_optimization.py`（默认使用强攻策略，可按需改参数）。
- 防御验证入口：`python scripts/run_defense_test.py`  
  - 防御示例：`--normal-preset critical --switch-penalty 0.5 --minority-boost 1.2`

### 4.4 复用图表与聚合

- 重新出图：`python scripts/visualize_results.py --run-dir <run_dir> --plots-dir <dest>`
- 汇总指标：`python scripts/comprehensive_analysis.py --run-dir <run_dir> --output <path>.json`

## 5. 数据与样本 / Data

- `data/commonsense_qa_validation.jsonl`：Commonsense QA 样本（可设 limit 抽样）。  
- `data/ethics_test.jsonl`：伦理类小样本（更开放，易触发从众）。  
- 示例小文件：`data/commonsense_qa_sampled_100.jsonl`、`data/ethics_selected_15.jsonl`。

## 6. 结果解读 / Interpreting Results

- **ASR**：最终共识被攻击目标取代的比例；**CR**：最初正确的正常代理被带偏的比例。  
- **Transition**：C→I 表示劣化（关注）、I→C 表示纠错。  
- **成本**：prompt/completion/total tokens，比较攻防开关的代价。  
- **显著性**：小样本 p 值不可靠，建议 50–100 题复核。

---

This document summarizes the current project structure, methods, and typical commands for running MAD conformity attack/defense experiments. Use the provided scripts/configs to reproduce Phase 1–4 experiments and extend them with custom factors.***
