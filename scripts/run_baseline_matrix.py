from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

from mad_debate.metrics import (
    build_answer_trajectories,
    compute_attack_success_rate,
    compute_conformity_rate,
    compute_transition_matrix,
    load_consensus_records,
    load_turn_logs,
    summarize_usage,
)
from scripts.run_analysis_demo import (
    build_config,
    perform_analysis,
    run_experiment,
)


SCENARIOS: Dict[str, Dict[str, str]] = {
    "A": {"attacker_model": "big", "follower_model": "big", "normal_model": "small"},
    "B": {"attacker_model": "small", "follower_model": "small", "normal_model": "big"},
    "C": {"attacker_model": "big", "follower_model": "big", "normal_model": "big"},
    "D": {"attacker_model": "small", "follower_model": "small", "normal_model": "small"},
    "E": {"attacker_model": "big", "follower_model": "small", "normal_model": "big"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline MAD scenarios batch and aggregate metrics.")
    parser.add_argument("--dataset", default="data/commonsense_qa_validation.jsonl")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--question-workers", type=int, default=2)
    parser.add_argument("--agent-workers", type=int, default=5)
    parser.add_argument("--consensus", choices=["majority", "confidence_weighted"], default="majority")
    parser.add_argument("--attacker-target", default="B")
    parser.add_argument("--normal-agents", type=int, default=3)
    parser.add_argument("--small-model", default="qwen3-8b")
    parser.add_argument("--big-model", default="qwen-max")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--log-dir", default="outputs")
    parser.add_argument("--plots-dir", default="outputs/plots")
    parser.add_argument("--scenarios", default="A,B,C,D,E", help="Comma-separated scenario labels to run")
    parser.add_argument("--follower-mode", choices=["independent_then_follow", "always_follow", "delayed_follow"], default="independent_then_follow")
    parser.add_argument("--attacker-tone", choices=["baseline", "assertive", "tentative", "authoritative"], default="baseline")
    parser.add_argument("--attacker-rationale", choices=["full", "minimal", "flawed"], default="full")
    parser.add_argument("--force-confidence", type=float, help="Force attacker confidence")
    parser.add_argument("--normal-preset", choices=["baseline", "conformist", "neutral", "critical", "independent"], default="baseline")
    parser.add_argument("--switch-penalty", type=float, default=1.0)
    parser.add_argument("--minority-boost", type=float, default=1.0)
    parser.add_argument("--output-summary", default="outputs/baseline_matrix_summary.json")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _make_namespace(base: argparse.Namespace, scenario_label: str, mapping: Dict[str, str]) -> SimpleNamespace:
    attacker_model = base.big_model if mapping["attacker_model"] == "big" else base.small_model
    follower_model = base.big_model if mapping["follower_model"] == "big" else base.small_model
    normal_model = base.big_model if mapping["normal_model"] == "big" else base.small_model
    return SimpleNamespace(
        dataset=base.dataset,
        dataset_type="commonsense_qa",
        limit=base.limit,
        shuffle=base.shuffle,
        seed=base.seed,
        rounds=base.rounds,
        question_workers=base.question_workers,
        agent_workers=base.agent_workers,
        consensus=base.consensus,
        attacker_target=base.attacker_target,
        normal_agents=base.normal_agents,
        model=base.small_model,
        attacker_model=attacker_model,
        follower_model=follower_model,
        normal_model=normal_model,
        temperature=base.temperature,
        max_tokens=base.max_tokens,
        log_dir=Path(base.log_dir) / scenario_label,
        plots_dir=Path(base.plots_dir) / scenario_label,
        follower_mode=base.follower_mode,
        attacker_tone=base.attacker_tone,
        attacker_rationale=base.attacker_rationale,
        force_confidence=base.force_confidence,
        normal_preset=base.normal_preset,
        switch_penalty=base.switch_penalty,
        minority_boost=base.minority_boost,
    )


def _summarize_run(run_dir: Path, config_args: SimpleNamespace) -> Dict[str, object]:
    log_path = run_dir / "debate_log.jsonl"
    turns = load_turn_logs(log_path)
    consensus = load_consensus_records(log_path)
    trajectories = build_answer_trajectories(turns)

    attacker_id = "attacker"
    follower_id = "follower"
    exclude_ids = {attacker_id, follower_id}

    conformity_stats = compute_conformity_rate(
        turns,
        attacker_id=attacker_id,
        follower_id=follower_id,
        reference="attacker",
        exclude_agent_ids=exclude_ids,
    )
    transition = compute_transition_matrix(
        trajectories,
        reference="ground_truth",
        exclude_agent_ids=exclude_ids,
    )
    attack_success = compute_attack_success_rate(trajectories, consensus)
    usage = summarize_usage(turns)
    acc_values = [c.is_correct for c in consensus if c.is_correct is not None]
    accuracy = sum(1 for v in acc_values if v) / len(acc_values) if acc_values else None

    return {
        "run_dir": str(run_dir),
        "dataset": config_args.dataset,
        "limit": config_args.limit,
        "rounds": config_args.rounds,
        "consensus": config_args.consensus,
        "models": {
            "attacker": config_args.attacker_model,
            "follower": config_args.follower_model,
            "normal": config_args.normal_model,
        },
        "attack_success": attack_success,
        "conformity": conformity_stats,
        "transition": {
            "c2c": transition.correct_to_correct,
            "c2i": transition.correct_to_incorrect,
            "i2c": transition.incorrect_to_correct,
            "i2i": transition.incorrect_to_incorrect,
        },
        "accuracy": accuracy,
        "usage": usage,
    }


def main() -> None:
    args = parse_args()
    selected = [s.strip().upper() for s in args.scenarios.split(",") if s.strip()]
    summaries: List[Dict[str, object]] = []
    for label in selected:
        if label not in SCENARIOS:
            print(f"Skipping unknown scenario {label}")
            continue
        scenario_args = _make_namespace(args, label, SCENARIOS[label])
        config = build_config(scenario_args)
        print(f"=== Running scenario {label} ===")
        run_experiment(config)
        perform_analysis(config, Path(scenario_args.plots_dir))
        if config.logging.run_dir:
            summaries.append({"scenario": label, **_summarize_run(config.logging.run_dir, scenario_args)})

    if summaries:
        output_path = Path(args.output_summary)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2, ensure_ascii=False)
        print(f"\nSaved aggregated summary to {output_path}")


if __name__ == "__main__":
    main()
