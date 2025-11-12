#!/usr/bin/env python
"""
Convert ETHICS commonsense CSV dataset to JSONL format for MAD framework.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mad_debate.datasets import load_ethics_dataset
import json


def main():
    parser = argparse.ArgumentParser(description="Convert ETHICS CSV to JSONL")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/ethics/data/commonsense/ambig.csv"),
        help="Path to ETHICS CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ethics_ambig.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--subset",
        choices=["test", "ambig", "test_hard"],
        default="ambig",
        help="Which ETHICS subset this is",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to convert",
    )
    args = parser.parse_args()

    print(f"Loading ETHICS dataset from: {args.input}")

    # Load dataset
    questions = load_ethics_dataset(args.input, subset=args.subset, config=None)

    # Apply limit if specified
    if args.limit:
        questions = questions[:args.limit]

    print(f"Loaded {len(questions)} questions")

    # Convert to JSONL format
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for q in questions:
            obj = {
                "id": q.question_id,
                "question": q.question,
                "choices": {
                    "label": [c.label for c in q.choices],
                    "text": [c.text for c in q.choices],
                },
                "answerKey": q.answer_key,
                "metadata": q.metadata,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved to: {args.output}")
    print(f"\nSample question:")
    print(f"  ID: {questions[0].question_id}")
    print(f"  Question: {questions[0].question[:150]}...")
    print(f"  Choices: {[c.label + ': ' + c.text for c in questions[0].choices]}")
    print(f"  Answer: {questions[0].answer_key}")


if __name__ == "__main__":
    main()
