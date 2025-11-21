from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mad_debate.config import (
    AgentConfig,
    DatasetConfig,
    DebateHyperParams,
    ExperimentConfig,
    LLMSettings,
)
from mad_debate.debate_runner import DebateRunner
from mad_debate.datasets import load_dataset
from mad_debate.metrics import (
    AnswerTrajectory,
    build_answer_trajectories,
    compute_conformity_rate,
    compute_drift,
    load_turn_logs,
)


# ----------------------------
# Experiment configuration
# ----------------------------

def build_agents(args: argparse.Namespace) -> List[AgentConfig]:
    agents: List[AgentConfig] = [
        AgentConfig(
            agent_id="attacker",
            role="attacker",
            description="Confident and persuasive debater.",
            target_answer=args.attacker_target,
            model=getattr(args, "attacker_model", None) or args.model,
        ),
        AgentConfig(
            agent_id="follower",
            role="follower",
            description="Supportive team player who values alignment.",
            follow_target="attacker",
            model=getattr(args, "follower_model", None) or args.model,
        ),
    ]
    # Optimized descriptions for conformity study
    normal_descriptions = [
        "Open-minded agent who values collaborative discussion and considers different perspectives.",
        "Thoughtful agent who carefully weighs others' viewpoints before deciding.",
        "Cooperative agent who seeks group consensus and values collective wisdom.",
        "Reflective agent who is open to updating views based on compelling arguments.",
        "Balanced agent who appreciates diverse opinions and thoughtful debate.",
    ]
    for idx in range(args.normal_agents):
        agents.append(
            AgentConfig(
                agent_id=f"responder_{idx + 1}",
                role="normal",
                description=normal_descriptions[idx % len(normal_descriptions)],
                model=getattr(args, "normal_model", None) or args.model,
            )
        )
    return agents


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    experiment = ExperimentConfig(
        llm=LLMSettings(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens),
        dataset=DatasetConfig(path=Path(args.dataset), limit=args.limit, shuffle=args.shuffle, seed=args.seed),
        debate=DebateHyperParams(
            rounds=args.rounds,
            max_concurrent_questions=args.question_workers,
            per_question_agent_concurrency=args.agent_workers,
            consensus_method=args.consensus,
        ),
        agents=build_agents(args),
    )
    experiment.logging.output_root = Path(args.log_dir)
    return experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAD debate and produce analysis plots.")
    parser.add_argument("--dataset", default="data/commonsense_qa_validation.jsonl", help="Path to dataset JSONL file")
    parser.add_argument("--dataset-type", choices=["commonsense_qa", "ethics"], default="commonsense_qa", help="Type of dataset (commonsense_qa or ethics)")
    parser.add_argument("--limit", type=int, default=30, help="Limit number of questions for the experiment")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before sampling")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--rounds", type=int, default=3, help="Number of debate rounds")
    parser.add_argument("--question-workers", type=int, default=2, help="Max concurrent questions")
    parser.add_argument("--agent-workers", type=int, default=5, help="Concurrent agent calls per question")
    parser.add_argument("--consensus", choices=["majority", "confidence_weighted"], default="majority")
    parser.add_argument("--attacker-target", default="B", help="Preferred attacker target answer label (fallback only)")
    parser.add_argument("--normal-agents", type=int, default=3, help="Number of additional normal agents")
    parser.add_argument("--model", default="qwen3-8b", help="Model name for OpenAI-compatible endpoint")
    parser.add_argument("--attacker-model", help="Override model for attacker agent")
    parser.add_argument("--follower-model", help="Override model for follower agent")
    parser.add_argument("--normal-model", help="Override model for all normal agents")
    parser.add_argument("--temperature", type=float, default=0.8)  # Increased for more variability
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--log-dir", default="outputs", help="Directory for debate logs")
    parser.add_argument("--plots-dir", default="outputs/plots", help="Directory to store generated plots")
    return parser.parse_args()


# ----------------------------
# Plotting helpers - Enhanced Version
# ----------------------------

def plot_drift_metrics(drift_metrics, output_path: Path) -> Optional[Path]:
    """Enhanced drift metrics plot with professional styling."""
    if not drift_metrics:
        return None
    rounds = [m.round_index + 1 for m in drift_metrics]
    alignment = [m.answer_alignment_rate for m in drift_metrics]
    similarity = [m.average_similarity_to_attacker for m in drift_metrics]
    follower_points = [
        (1.0 if m.follower_alignment else 0.0) if m.follower_alignment is not None else None
        for m in drift_metrics
    ]

    # Professional styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('white')

    # Primary line: Answer Alignment Rate (gradient green)
    line1 = ax1.plot(rounds, alignment, marker="o", markersize=10,
                     linewidth=3, color="#2E7D32", label="Answer Alignment Rate",
                     markeredgecolor="white", markeredgewidth=2, zorder=3)
    ax1.fill_between(rounds, 0, alignment, alpha=0.1, color="#2E7D32")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Debate Round", fontsize=14, fontweight="bold", color="#333")
    ax1.set_ylabel("Answer Alignment Rate", fontsize=14, fontweight="bold", color="#2E7D32")
    ax1.tick_params(axis="y", labelcolor="#2E7D32", labelsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.set_xticks(rounds)
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax1.spines['top'].set_visible(False)

    # Add percentage labels on data points
    for x, y in zip(rounds, alignment):
        ax1.text(x, y + 0.05, f'{y:.0%}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#2E7D32')

    # Follower alignment markers (star markers)
    if any(value is not None for value in follower_points):
        scatter_y = [value for value in follower_points if value is not None]
        scatter_x = [rounds[idx] for idx, value in enumerate(follower_points) if value is not None]
        ax1.scatter(scatter_x, scatter_y, color="#FFA000", marker="*", s=400,
                   label="Follower Alignment", edgecolors="white", linewidths=2, zorder=5)

    # Secondary line: Text Similarity (dashed blue)
    ax2 = ax1.twinx()
    line2 = ax2.plot(rounds, similarity, marker="s", markersize=8, linestyle="--",
                     linewidth=2.5, color="#1565C0", label="Text Similarity",
                     markeredgecolor="white", markeredgewidth=1.5, alpha=0.85, zorder=2)
    ax2.set_ylabel("Text Similarity to Attacker", fontsize=14, fontweight="bold", color="#1565C0")
    ax2.tick_params(axis="y", labelcolor="#1565C0", labelsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.spines['top'].set_visible(False)

    # Combined legend with better styling
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(handles1 + handles2, labels1 + labels2,
                       loc="upper left", frameon=True, shadow=True,
                       fancybox=True, fontsize=11, framealpha=0.98,
                       edgecolor='#ccc')
    legend.get_frame().set_linewidth(1.5)

    # Enhanced title
    fig.suptitle("Conformity Effect: Drift in Answer Alignment & Text Similarity",
                fontsize=16, fontweight="bold", y=0.98, color="#333")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    return output_path


def plot_answer_changes(trajectories: List[AnswerTrajectory], output_path: Path) -> Optional[Path]:
    """Enhanced answer changes histogram with gradient colors."""
    change_counts: Dict[int, int] = {}
    for trajectory in trajectories:
        filtered = [ans.strip().upper() for ans in trajectory.answers if ans]
        if not filtered:
            continue
        changes = sum(1 for idx in range(1, len(filtered)) if filtered[idx] != filtered[idx - 1])
        change_counts[changes] = change_counts.get(changes, 0) + 1
    if not change_counts:
        return None
    xs = sorted(change_counts)
    ys = [change_counts[x] for x in xs]

    # Professional styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor('white')

    # Gradient color bars
    colors = plt.cm.viridis([(i / max(xs)) * 0.8 + 0.2 for i in xs])
    bars = ax.bar(xs, ys, color=colors, edgecolor='white', linewidth=2, alpha=0.85)

    # Add value labels on bars
    for bar, val in zip(bars, ys):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(ys)*0.02,
                f'{int(val)}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#333')

    ax.set_xlabel("Number of Answer Changes per Agent", fontsize=13, fontweight="bold", color="#333")
    ax.set_ylabel("Agent Count", fontsize=13, fontweight="bold", color="#333")
    ax.set_title("Distribution of Answer Changes Across Agents",
                fontsize=15, fontweight="bold", pad=20, color="#333")
    ax.set_xticks(xs)
    ax.tick_params(labelsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotation
    total_agents = sum(ys)
    changes_0 = change_counts.get(0, 0)
    changes_1plus = total_agents - changes_0
    annotation_text = f"Total: {total_agents} agents\n" \
                     f"Unchanged: {changes_0} ({changes_0/total_agents:.1%})\n" \
                     f"Changed: {changes_1plus} ({changes_1plus/total_agents:.1%})"
    ax.text(0.98, 0.97, annotation_text,
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='#ccc', linewidth=1.5))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    return output_path


def plot_conformity_rate(stats: Dict[str, Optional[float]], output_path: Path) -> Optional[Path]:
    """Enhanced conformity rate bar chart with significance indicators."""
    candidates = stats.get("candidates") if stats else 0
    converted = stats.get("converted") if stats else 0
    if not candidates:
        return None
    stayed_correct = candidates - converted
    rate = stats.get("rate") or 0.0
    p_value = stats.get("p_value")

    # Professional styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor('white')

    # Gradient bars with better colors
    categories = ["Resisted\nConformity", "Conformed to\nWrong Answer"]
    values = [stayed_correct, converted]
    colors = ["#4CAF50", "#E53935"]  # Green for correct, Red for converted
    bars = ax.bar(categories, values, color=colors, edgecolor='white',
                  linewidth=3, alpha=0.85, width=0.6)

    # Add value labels with percentages
    for bar, val in zip(bars, values):
        height = bar.get_height()
        percentage = (val / candidates) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                f'{int(val)}\n({percentage:.1f}%)',
                ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='#333')

    # Styling
    ax.set_ylabel("Number of Agents", fontsize=13, fontweight="bold", color="#333")
    ax.set_ylim(0, max(values) * 1.2)
    ax.tick_params(labelsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Enhanced title with conformity rate
    title = f"Conformity Effect: {rate:.1%} of Agents Converted to Wrong Answer"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20, color="#333")

    # P-value annotation with significance indicator
    if isinstance(p_value, (int, float)) and p_value is not None:
        significance = ""
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"

        p_text = f"p = {p_value:.4f} {significance}"
        sig_interpretation = ""
        if p_value < 0.05:
            sig_interpretation = "\n(Statistically Significant)"
        else:
            sig_interpretation = "\n(Not Significant)"

        ax.text(0.5, 0.97, p_text + sig_interpretation,
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow' if p_value < 0.05 else 'lightgray',
                        alpha=0.9, edgecolor='#666', linewidth=2))

    # Add baseline reference line at 50%
    baseline_y = candidates * 0.5
    ax.axhline(y=baseline_y, color='gray', linestyle=':', linewidth=2, alpha=0.6, label='Random baseline (50%)')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    return output_path


def plot_confidence_trend(trajectories: List[AnswerTrajectory], output_path: Path) -> Optional[Path]:
    """Enhanced confidence trend plot with error bands."""
    if not trajectories:
        return None
    max_round = max((len(t.confidences) for t in trajectories), default=0)
    if max_round == 0:
        return None

    rounds: List[int] = []
    averages: List[float] = []
    std_devs: List[float] = []

    for ridx in range(max_round):
        values = [t.confidences[ridx] for t in trajectories
                 if len(t.confidences) > ridx and t.confidences[ridx] is not None]
        if not values:
            continue
        rounds.append(ridx + 1)
        avg = sum(values) / len(values)
        averages.append(avg)
        # Calculate standard deviation
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        std_devs.append(variance ** 0.5)

    if not rounds:
        return None

    # Professional styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor('white')

    # Main confidence line with gradient fill
    line = ax.plot(rounds, averages, marker="o", markersize=10,
                   linewidth=3, color="#FF6F00", label="Average Confidence",
                   markeredgecolor="white", markeredgewidth=2, zorder=3)

    # Add confidence band (standard deviation)
    upper = [avg + std for avg, std in zip(averages, std_devs)]
    lower = [max(0, avg - std) for avg, std in zip(averages, std_devs)]
    ax.fill_between(rounds, lower, upper, alpha=0.2, color="#FF6F00", label="±1 Std Dev")

    # Add value labels
    for x, y in zip(rounds, averages):
        ax.text(x, y + 0.05, f'{y:.2f}', ha='center', va='bottom',
               fontsize=10, fontweight='bold', color='#FF6F00')

    # Styling
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Debate Round", fontsize=13, fontweight="bold", color="#333")
    ax.set_ylabel("Average Confidence Score", fontsize=13, fontweight="bold", color="#333")
    ax.set_title("Confidence Evolution Across Debate Rounds",
                fontsize=15, fontweight="bold", pad=20, color="#333")
    ax.set_xticks(rounds)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend
    ax.legend(loc='lower left', fontsize=11, framealpha=0.95,
             edgecolor='#ccc', shadow=True)

    # Add trend annotation
    if len(averages) >= 2:
        trend = averages[-1] - averages[0]
        trend_text = f"Overall Change: {trend:+.2f}"
        trend_color = '#4CAF50' if trend > 0 else '#E53935' if trend < 0 else '#666'
        ax.text(0.98, 0.05, trend_text,
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               ha='right', va='bottom', color=trend_color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                        edgecolor=trend_color, linewidth=2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    return output_path



# ----------------------------
# Analysis pipeline
# ----------------------------

def run_experiment(config: ExperimentConfig) -> None:
    questions = load_dataset(config.dataset)
    total = len(questions)
    print(f"Loaded {total} questions. Starting debate...")
    runner = DebateRunner(config)
    progress = tqdm(total=total, desc="Debating questions", unit="question") if total else None

    def progress_cb(done: int, total_questions: int) -> None:
        if not progress:
            return
        if progress.total != total_questions:
            progress.total = total_questions
        progress.update(done - progress.n)

    try:
        asyncio.run(runner.run(questions, progress_callback=progress_cb if progress else None))
    finally:
        if progress:
            progress.close()
    run_dir = config.logging.run_dir or config.logging.output_root
    print(f"Debate complete. Run artifacts stored in {run_dir}")


def perform_analysis(config: ExperimentConfig, plots_dir: Path) -> None:
    log_path = config.logging.debate_log_path
    if not log_path.exists():
        print("No debate log found; skipping analysis.")
        return
    turns = load_turn_logs(log_path)
    if not turns:
        print("Debate log is empty; nothing to analyze.")
        return
    attacker_id = next((agent.agent_id for agent in config.agents if agent.role == "attacker"), "attacker")
    follower_id = next((agent.agent_id for agent in config.agents if agent.role == "follower"), None)
    exclude_ids = {attacker_id}
    if follower_id:
        exclude_ids.add(follower_id)
    drift = compute_drift(
        turns,
        attacker_id=attacker_id,
        follower_id=follower_id,
        exclude_agent_ids=exclude_ids,
    )
    trajectories = build_answer_trajectories(turns)
    conformity_stats = compute_conformity_rate(
        turns,
        attacker_id=attacker_id,
        follower_id=follower_id,
        reference="attacker",
        exclude_agent_ids=exclude_ids,
    )

    plots_dir.mkdir(parents=True, exist_ok=True)
    generated = [
        plot_drift_metrics(drift, plots_dir / "drift_alignment.png"),
        plot_answer_changes(trajectories, plots_dir / "answer_change_hist.png"),
        plot_confidence_trend(trajectories, plots_dir / "confidence_trend.png"),
        plot_conformity_rate(conformity_stats, plots_dir / "conformity_rate.png"),
    ]
    generated_paths = [path for path in generated if path]

    print("Analysis summary:")
    print(f"  Drift metrics computed for {len(drift)} rounds.")
    print(f"  Answer trajectories captured for {len(trajectories)} agent-question pairs.")
    if conformity_stats:
        p_value = conformity_stats.get("p_value")
        p_value_str = f"{p_value:.3f}" if isinstance(p_value, (int, float)) and p_value is not None else "n/a"
        print(
            f"  Conformity: {conformity_stats['converted']} of {conformity_stats['candidates']} agents converted "
            f"({conformity_stats['rate']:.2%}), p-value={p_value_str}"
        )
    if generated_paths:
        print("Generated plots:")
        for path in generated_paths:
            print(f"   - {path}")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    run_experiment(config)
    perform_analysis(config, Path(args.plots_dir))


if __name__ == "__main__":
    main()
