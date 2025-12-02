#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run Phase 1 baseline scenarios A-E with defaults.
python scripts/run_baseline_matrix.py \
  --dataset data/commonsense_qa_validation.jsonl \
  --limit 30 \
  --rounds 3 \
  --attacker-target B \
  --small-model qwen3-8b \
  --big-model qwen-max \
  --scenarios A,B,C,D,E "$@"
