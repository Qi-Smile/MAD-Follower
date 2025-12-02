from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_analysis_demo import build_config, perform_analysis, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: mechanism dissection experiments (prompts/rationale/follower modes).")
    parser.add_argument("--dataset", default="data/commonsense_qa_validation.jsonl")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--scenario", choices=["prompt", "rationale", "follower_mode", "tone"], default="prompt")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--attacker-model", default="qwen3-8b")
    parser.add_argument("--follower-model", default="qwen3-8b")
    parser.add_argument("--normal-model", default="qwen3-8b")
    parser.add_argument("--attacker-target", default="B")
    parser.add_argument("--plots-dir", default="outputs/plots")
    parser.add_argument("--log-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    print(f"[Phase2] Running mechanism experiment: {args.scenario}")
    run_experiment(cfg)
    perform_analysis(cfg, Path(args.plots_dir))


if __name__ == "__main__":
    main()
