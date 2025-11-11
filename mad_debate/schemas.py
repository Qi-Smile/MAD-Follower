from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.utcnow()


@dataclass(slots=True)
class AgentTurnRecord:
    question_id: str
    round_index: int
    agent_id: str
    role: str
    prompt: str
    context_snippet: str
    response_text: str
    parsed_answer: Optional[str]
    rationale: Optional[str]
    confidence: Optional[float]
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "question_id": self.question_id,
            "round_index": self.round_index,
            "agent_id": self.agent_id,
            "role": self.role,
            "prompt": self.prompt,
            "context_snippet": self.context_snippet,
            "response_text": self.response_text,
            "parsed_answer": self.parsed_answer,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentTurnRecord":
        return cls(
            question_id=data["question_id"],
            round_index=int(data["round_index"]),
            agent_id=data["agent_id"],
            role=data.get("role", ""),
            prompt=data.get("prompt", ""),
            context_snippet=data.get("context_snippet", ""),
            response_text=data.get("response_text", ""),
            parsed_answer=data.get("parsed_answer"),
            rationale=data.get("rationale"),
            confidence=data.get("confidence"),
            latency_ms=float(data.get("latency_ms", 0.0)),
            timestamp=_parse_datetime(data.get("timestamp")),
            metadata=data.get("metadata", {}) or {},
        )


@dataclass(slots=True)
class ConsensusRecord:
    question_id: str
    method: str
    final_answer: Optional[str]
    supporting_agents: List[str] = field(default_factory=list)
    votes: Dict[str, int] = field(default_factory=dict)
    confidence: Optional[float] = None
    is_correct: Optional[bool] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "method": self.method,
            "final_answer": self.final_answer,
            "supporting_agents": self.supporting_agents,
            "votes": self.votes,
            "confidence": self.confidence,
            "is_correct": self.is_correct,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(slots=True)
class RunSummary:
    question_count: int
    attacker_id: Optional[str]
    follower_id: Optional[str]
    accuracy: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_count": self.question_count,
            "attacker_id": self.attacker_id,
            "follower_id": self.follower_id,
            "accuracy": self.accuracy,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
