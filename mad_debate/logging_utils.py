from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .schemas import AgentTurnRecord, ConsensusRecord, RunSummary


class ExperimentLogger:
    """Thread-safe logger that stores run artifacts in a dedicated directory."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.debate_log_path = run_dir / "debate_log.jsonl"
        self.summary_path = run_dir / "run_summary.json"
        self.config_path = run_dir / "config.json"
        self.questions_dir = run_dir / "questions"
        self.questions_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._question_index = 0

    def log_turn(self, record: AgentTurnRecord) -> None:
        payload = record.to_dict()
        self._write_jsonl([payload])

    def log_turns(self, records: Iterable[AgentTurnRecord]) -> None:
        payloads = [record.to_dict() for record in records]
        self._write_jsonl(payloads)

    def _write_jsonl(self, payloads: List[dict]) -> None:
        if not payloads:
            return
        with self._lock:
            with self.debate_log_path.open("a", encoding="utf-8") as handle:
                for payload in payloads:
                    handle.write(json.dumps(payload, ensure_ascii=False))
                    handle.write("\n")

    def log_consensus(self, record: ConsensusRecord) -> None:
        with self._lock:
            with self.debate_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"consensus": record.to_dict()}, ensure_ascii=False))
                handle.write("\n")

    def log_warning(self, payload: dict) -> None:
        with self._lock:
            with self.debate_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"warning": payload}, ensure_ascii=False))
                handle.write("\n")

    def write_summary(self, summary: RunSummary) -> None:
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary.to_dict(), handle, indent=2, ensure_ascii=False)

    def write_config(self, payload: Dict[str, Any]) -> None:
        with self.config_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def log_question_detail(
        self,
        question_id: str,
        payload: Dict[str, Any],
    ) -> Path:
        self._question_index += 1
        slug = self._slugify(question_id)
        filename = f"{self._question_index:03d}_{slug}.json"
        path = self.questions_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def _slugify(text: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
        return cleaned.strip("_") or "question"
