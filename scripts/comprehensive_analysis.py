from __future__ import annotations

import argparse
from pathlib import Path
import json

from mad_debate.analysis import summarize_core_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate metrics from an existing run directory.")
    parser.add_argument("--run-dir", required=True, help="Path to run directory containing debate_log.jsonl")
    parser.add_argument("--attacker-id", default="attacker")
    parser.add_argument("--follower-id", default="follower")
    parser.add_argument("--output", help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.run_dir) / "debate_log.jsonl"
    metrics = summarize_core_metrics(
        log_path,
        attacker_id=args.attacker_id,
        follower_id=args.follower_id,
    )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False, default=str)
        print(f"Saved aggregated metrics to {out}")
    else:
        print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
