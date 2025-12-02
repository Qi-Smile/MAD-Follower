from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_analysis_demo import perform_analysis, build_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate plots for an existing run.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing debate_log.jsonl")
    parser.add_argument("--plots-dir", required=True, help="Destination directory for plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Dummy config that points logging to existing run_dir
    cfg = build_config(argparse.Namespace(
        dataset="",
        dataset_type="commonsense_qa",
        limit=0,
        shuffle=False,
        seed=42,
        rounds=3,
        question_workers=1,
        agent_workers=1,
        consensus="majority",
        attacker_target="B",
        normal_agents=3,
        model="qwen3-8b",
        attacker_model=None,
        follower_model=None,
        normal_model=None,
        temperature=0.8,
        max_tokens=512,
        log_dir=args.run_dir,
        plots_dir=args.plots_dir,
        follower_mode="independent_then_follow",
        attacker_tone="baseline",
        attacker_rationale="full",
        force_confidence=None,
        normal_preset="baseline",
        switch_penalty=1.0,
        minority_boost=1.0,
    ))
    cfg.logging.run_dir = Path(args.run_dir)
    perform_analysis(cfg, Path(args.plots_dir))


if __name__ == "__main__":
    main()
