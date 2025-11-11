from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .config import DatasetConfig


@dataclass(slots=True)
class CommonsenseChoice:
    label: str
    text: str


@dataclass(slots=True)
class CommonsenseQuestion:
    question_id: str
    question: str
    choices: List[CommonsenseChoice]
    answer_key: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def choice_text(self, label: str) -> Optional[str]:
        for choice in self.choices:
            if choice.label == label:
                return choice.text
        return None


def _parse_choices(raw_choices: Any) -> List[CommonsenseChoice]:
    choices: List[CommonsenseChoice] = []
    if isinstance(raw_choices, dict):
        labels = raw_choices.get("label")
        texts = raw_choices.get("text")
        if isinstance(labels, list) and isinstance(texts, list) and len(labels) == len(texts):
            for label, text in zip(labels, texts):
                choices.append(CommonsenseChoice(label=str(label), text=str(text)))
            return choices
        iterable: Iterable = raw_choices.items()
    else:
        iterable = raw_choices
    for entry in iterable:
        if isinstance(entry, tuple):
            label, text = entry
        else:
            label = entry.get("label") or entry.get("key")
            text = entry.get("text") or entry.get("answer")
        choices.append(CommonsenseChoice(label=str(label), text=str(text)))
    return choices


def load_dataset(config: DatasetConfig) -> List[CommonsenseQuestion]:
    """Loads CommonsenseQA-style questions from a JSONL file."""

    path: Path = config.path
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    questions: List[CommonsenseQuestion] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            qid = data.get("id") or data.get("question_id")
            question = data.get("question") or data.get("stem")
            if not qid or not question:
                continue
            choices = _parse_choices(data.get("choices") or data.get("options"))
            answer_key = data.get("answerKey") or data.get("answer")
            metadata = {k: v for k, v in data.items() if k not in {"id", "question", "stem", "choices", "options", "answerKey", "answer"}}
            questions.append(
                CommonsenseQuestion(
                    question_id=str(qid),
                    question=str(question),
                    choices=choices,
                    answer_key=str(answer_key) if answer_key is not None else None,
                    metadata=metadata,
                )
            )

    if config.shuffle:
        random.Random(config.seed).shuffle(questions)

    if config.limit is not None:
        questions = questions[: config.limit]

    return questions
