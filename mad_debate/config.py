from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

AgentRole = Literal["attacker", "follower", "normal", "judge"]


@dataclass(slots=True)
class LLMSettings:
    """LLM-related configuration shared across agents."""

    model: str = "qwen3-8b"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout_seconds: int = 60
    max_concurrent_requests: int = 6
    max_retries: int = 2


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for loading CommonsenseQA-style datasets."""

    path: Path = Path("data/commonsense_qa_validation.jsonl")
    limit: Optional[int] = None
    shuffle: bool = False
    seed: int = 42


@dataclass(slots=True)
class LoggingConfig:
    """Output settings for experiment artifacts."""

    output_root: Path = Path("outputs")
    run_dir: Optional[Path] = None

    def ensure_root(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, descriptor: str) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_descriptor = descriptor.replace(" ", "_")
        dir_name = f"{timestamp}_{safe_descriptor}"
        run_dir = self.output_root / dir_name
        run_dir.mkdir(parents=True, exist_ok=False)
        self.run_dir = run_dir
        return run_dir

    @property
    def debate_log_path(self) -> Path:
        if not self.run_dir:
            raise RuntimeError("Run directory not initialized.")
        return self.run_dir / "debate_log.jsonl"

    @property
    def summary_path(self) -> Path:
        if not self.run_dir:
            raise RuntimeError("Run directory not initialized.")
        return self.run_dir / "run_summary.json"

    @property
    def config_path(self) -> Path:
        if not self.run_dir:
            raise RuntimeError("Run directory not initialized.")
        return self.run_dir / "config.json"

    @property
    def questions_dir(self) -> Path:
        if not self.run_dir:
            raise RuntimeError("Run directory not initialized.")
        path = self.run_dir / "questions"
        path.mkdir(exist_ok=True)
        return path


@dataclass(slots=True)
class AgentConfig:
    """Describes a single agent participating in the debate."""

    agent_id: str
    role: AgentRole
    description: str = ""
    target_answer: Optional[str] = None
    follow_target: Optional[str] = None
    model: Optional[str] = None


@dataclass(slots=True)
class AttackSettings:
    """Attack-side strategy knobs."""

    follower_mode: Literal["independent_then_follow", "always_follow", "delayed_follow"] = "independent_then_follow"
    attacker_tone: Literal["baseline", "assertive", "tentative", "authoritative"] = "baseline"
    rationale_style: Literal["full", "minimal", "flawed"] = "full"
    force_confidence: Optional[float] = None


@dataclass(slots=True)
class DefenseSettings:
    """Defense-side strategy knobs."""

    normal_prompt_preset: Literal["baseline", "conformist", "neutral", "critical", "independent"] = "baseline"
    confidence_penalty_for_switchers: float = 1.0
    minority_boost: float = 1.0


@dataclass(slots=True)
class DebateHyperParams:
    """Hyper-parameters that define how debates are executed."""

    rounds: int = 3
    max_concurrent_questions: int = 2
    per_question_agent_concurrency: int = 4
    consensus_method: Literal["majority", "confidence_weighted"] = "majority"


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level configuration container for a MAD experiment."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    attack: AttackSettings = field(default_factory=AttackSettings)
    defense: DefenseSettings = field(default_factory=DefenseSettings)
    debate: DebateHyperParams = field(default_factory=DebateHyperParams)
    agents: List[AgentConfig] = field(default_factory=list)

    def ensure_output_dir(self) -> None:
        self.logging.ensure_root()
