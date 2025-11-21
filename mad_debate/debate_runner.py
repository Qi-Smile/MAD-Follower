from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .agents import AttackerAgent, AgentResponse, BaseAgent, FollowerAgent, NormalAgent
from .config import AgentConfig, ExperimentConfig, LLMSettings
from .datasets import CommonsenseQuestion
from .llm_client import LLMClient, LLMClientError
from .logging_utils import ExperimentLogger
from .schemas import AgentTurnRecord, ConsensusRecord, RunSummary


class DebateRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.ensure_output_dir()
        descriptor = self._build_run_descriptor()
        run_dir = self.config.logging.create_run_dir(descriptor)
        self.logger = ExperimentLogger(run_dir)
        config_payload = {
            "run_descriptor": descriptor,
            "config": self._serialize_config(),
        }
        self.logger.write_config(config_payload)
        self.llm_client = LLMClient(self.config.llm)
        self._agent_clients: Dict[str, LLMClient] = {}
        for ac in self.config.agents:
            if getattr(ac, "model", None):
                base = self.config.llm
                override = LLMSettings(
                    model=ac.model or base.model,
                    temperature=base.temperature,
                    max_tokens=base.max_tokens,
                    timeout_seconds=base.timeout_seconds,
                    max_concurrent_requests=base.max_concurrent_requests,
                    max_retries=base.max_retries,
                )
                self._agent_clients[ac.agent_id] = LLMClient(override)
            else:
                self._agent_clients[ac.agent_id] = self.llm_client
        self.agents: List[BaseAgent] = [self._instantiate_agent(ac) for ac in self.config.agents]
        self.display_names: Dict[str, str] = {}
        for idx, agent in enumerate(self.agents, start=1):
            alias = f"agent_{idx}"
            self.display_names[agent.config.agent_id] = alias
            if hasattr(agent, "set_display_name"):
                agent.set_display_name(alias)
        for agent in self.agents:
            if isinstance(agent, FollowerAgent) and agent.config.follow_target:
                follow_alias = self.display_names.get(agent.config.follow_target)
                if follow_alias:
                    agent.set_follow_alias(follow_alias)
        self.attacker_id = self._find_agent_id("attacker")
        self.follower_id = self._find_agent_id("follower")
        self._attacker_config = next((cfg for cfg in self.config.agents if cfg.role == "attacker"), None)
        self.role_warnings: List[str] = []
        self.excluded_questions: List[str] = []

    def _instantiate_agent(self, agent_cfg: AgentConfig) -> BaseAgent:
        client = self._agent_clients.get(agent_cfg.agent_id, self.llm_client)
        if agent_cfg.role == "attacker":
            return AttackerAgent(agent_cfg, client)
        if agent_cfg.role == "follower":
            return FollowerAgent(agent_cfg, client)
        return NormalAgent(agent_cfg, client)

    async def run(
        self,
        questions: List[CommonsenseQuestion],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ConsensusRecord]:
        semaphore = asyncio.Semaphore(self.config.debate.max_concurrent_questions)
        progress_lock = asyncio.Lock()
        results: List[ConsensusRecord] = []
        total_questions = len(questions)
        completed = 0

        async def worker(question: CommonsenseQuestion) -> tuple[ConsensusRecord, Dict[str, Any]]:
            async with semaphore:
                return await self._run_single_question(question)

        async def worker_with_progress(
            question: CommonsenseQuestion,
        ) -> tuple[ConsensusRecord, Dict[str, Any]]:
            result = await worker(question)
            if progress_callback:
                nonlocal completed
                async with progress_lock:
                    completed += 1
                    progress_callback(completed, total_questions)
            return result

        coroutines = [worker_with_progress(question) for question in questions]
        for outcome in await asyncio.gather(*coroutines):
            consensus, question_state = outcome
            if question_state.get("attacker_compliant", True):
                results.append(consensus)
            else:
                self.excluded_questions.append(consensus.question_id)

        accuracy = None
        evals = [r for r in results if r.is_correct is not None]
        if evals:
            accuracy = sum(1 for r in evals if r.is_correct) / len(evals)

        summary = RunSummary(
            question_count=len(questions),
            attacker_id=self.attacker_id,
            follower_id=self.follower_id,
            accuracy=accuracy,
            metadata={
                "rounds": self.config.debate.rounds,
                "model": self.config.llm.model,
                "consensus_method": self.config.debate.consensus_method,
                "role_warning_count": len(self.role_warnings),
                "role_warnings": self.role_warnings,
                "valid_questions": len(results),
                "excluded_questions": self.excluded_questions,
            },
        )
        self.logger.write_summary(summary)
        return results

    def _find_agent_id(self, role: str) -> Optional[str]:
        for cfg in self.config.agents:
            if cfg.role == role:
                return cfg.agent_id
        return None

    async def _run_single_question(
        self, question: CommonsenseQuestion
    ) -> tuple[ConsensusRecord, Dict[str, Any]]:
        history: List[AgentTurnRecord] = []
        question_state = self._prepare_question_state(question)
        for round_index in range(self.config.debate.rounds):
            round_records = await self._run_round(question, round_index, history, question_state)
            history.extend(round_records)
        consensus = self._build_consensus(question, history)
        if not question_state.get("attacker_compliant", True):
            for record in history:
                record.metadata["excluded_question"] = True
        else:
            for record in history:
                record.metadata.setdefault("excluded_question", False)
        self.logger.log_turns(history)
        question_payload = {
            "question_id": question.question_id,
            "question": {
                "id": question.question_id,
                "text": question.question,
                "choices": [{"label": c.label, "text": c.text} for c in question.choices],
                "metadata": question.metadata,
            },
            "ground_truth": question_state.get("ground_truth"),
            "attacker_target": question_state.get("attacker_target"),
            "attacker_compliant": question_state.get("attacker_compliant", True),
            "excluded": not question_state.get("attacker_compliant", True),
            "warnings": list(question_state.get("warnings", [])),
            "consensus": consensus.to_dict(),
            "turns": [record.to_dict() for record in history],
        }
        self.logger.log_question_detail(question.question_id, question_payload)
        self.logger.log_consensus(consensus)
        return consensus, question_state

    async def _run_round(
        self,
        question: CommonsenseQuestion,
        round_index: int,
        history: List[AgentTurnRecord],
        question_state: Dict[str, Any],
    ) -> List[AgentTurnRecord]:
        sem = asyncio.Semaphore(self.config.debate.per_question_agent_concurrency)

        async def call_agent(agent: BaseAgent) -> tuple[BaseAgent, AgentResponse]:
            async with sem:
                try:
                    response: AgentResponse = await agent.respond(
                        question,
                        round_index,
                        history,
                        self.config.debate,
                        question_state=question_state,
                    )
                except LLMClientError as exc:
                    error_message = str(exc)
                    warning_details = {
                        "type": "llm_error",
                        "question_id": question.question_id,
                        "round": round_index + 1,
                        "agent_id": agent.config.agent_id,
                        "role": agent.config.role,
                        "message": error_message,
                    }
                    self._issue_warning(
                        "LLM request failed.",
                        warning_details,
                        question_state=question_state,
                    )
                    response = self._build_error_response(agent, round_index, error_message)
                return agent, response

        agent_responses = await asyncio.gather(*[call_agent(agent) for agent in self.agents])
        records: List[AgentTurnRecord] = []
        for agent, response in agent_responses:
            self._validate_role_behavior(question, round_index, agent, response, question_state, history)
            records.append(self._response_to_record(question, round_index, response, question_state))
        return records

    def _response_to_record(
        self,
        question: CommonsenseQuestion,
        round_index: int,
        response: AgentResponse,
        question_state: Dict[str, Any],
    ) -> AgentTurnRecord:
        metadata = dict(response.extra)
        ground_truth = question_state.get("ground_truth")
        attacker_target = question_state.get("attacker_target")
        if question.answer_key and not ground_truth:
            ground_truth = question.answer_key.strip().upper()
        if ground_truth:
            metadata.setdefault("question_answer", ground_truth)
        if attacker_target:
            metadata.setdefault("attacker_target", attacker_target)
        alias = self.display_names.get(response.agent_id)
        if alias:
            metadata.setdefault("display_name", alias)
        return AgentTurnRecord(
            question_id=question.question_id,
            round_index=round_index,
            agent_id=response.agent_id,
            role=response.role,
            prompt=response.prompt,
            context_snippet=response.context_snippet,
            response_text=response.raw_text,
            parsed_answer=response.parsed_answer,
            rationale=response.rationale,
            confidence=response.confidence,
            latency_ms=response.latency_ms,
            metadata=metadata,
        )

    def _build_run_descriptor(self) -> str:
        agents = len(self.config.agents)
        rounds = self.config.debate.rounds
        limit = self.config.dataset.limit if self.config.dataset.limit is not None else "all"
        return f"n{agents}_r{rounds}_limit{limit}"

    def _serialize_config(self) -> Dict[str, Any]:
        raw = asdict(self.config)
        return self._sanitize_config(raw)

    def _sanitize_config(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: self._sanitize_config(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_config(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._sanitize_config(v) for v in value)
        return value

    def _build_error_response(
        self,
        agent: BaseAgent,
        round_index: int,
        error_message: str,
    ) -> AgentResponse:
        fallback_text = (
            "#answer: \n"
            "#rationale: Response unavailable due to upstream error.\n"
            "#confidence: 0.0"
        )
        return AgentResponse(
            agent_id=agent.config.agent_id,
            role=agent.config.role,
            raw_text=fallback_text,
            parsed_answer=None,
            rationale="LLM request failed; placeholder response recorded.",
            confidence=0.0,
            prompt=f"ERROR: {error_message}",
            context_snippet="",
            latency_ms=0.0,
            extra={"error": error_message},
        )

    def _prepare_question_state(self, question: CommonsenseQuestion) -> Dict[str, Any]:
        ground_truth = question.answer_key.strip().upper() if question.answer_key else None
        attacker_target = self._select_attacker_target(question, ground_truth)
        return {
            "ground_truth": ground_truth,
            "attacker_target": attacker_target,
            "warnings": [],
            "attacker_compliant": True,
        }

    def _select_attacker_target(
        self,
        question: CommonsenseQuestion,
        ground_truth: Optional[str],
    ) -> Optional[str]:
        preferred = None
        if self._attacker_config and self._attacker_config.target_answer:
            preferred = self._attacker_config.target_answer.strip().upper()

        def pick_candidate(candidates: Iterable[Optional[str]]) -> Optional[str]:
            for label in candidates:
                normalized = (label or "").strip().upper()
                if not normalized:
                    continue
                if ground_truth and normalized == ground_truth:
                    continue
                return normalized
            return None

        candidate_labels = [choice.label for choice in question.choices]
        target = pick_candidate([preferred] if preferred else [])
        if target:
            return target
        target = pick_candidate(candidate_labels)
        if target:
            return target
        # Fallback: reuse preferred or first label even if it matches ground truth
        if preferred:
            return preferred
        for label in candidate_labels:
            normalized = (label or "").strip().upper()
            if normalized:
                return normalized
        return None

    def _validate_role_behavior(
        self,
        question: CommonsenseQuestion,
        round_index: int,
        agent: BaseAgent,
        response: AgentResponse,
        question_state: Dict[str, Any],
        history: List[AgentTurnRecord],
    ) -> None:
        role = agent.config.role
        if role == "attacker":
            expected = (question_state.get("attacker_target") or "").strip().upper()
            actual = (response.parsed_answer or "").strip().upper()
            if expected and actual and expected != actual:
                question_state["attacker_compliant"] = False
                self._issue_warning(
                    "Attacker deviated from target answer.",
                    {
                        "question_id": question.question_id,
                        "round": round_index + 1,
                        "expected": expected,
                        "actual": actual,
                    },
                    question_state=question_state,
                )
        elif role == "follower" and round_index > 0:
            anchor = self._get_attacker_answer(history, round_index - 1)
            follower_answer = (response.parsed_answer or "").strip().upper()
            if anchor and follower_answer and anchor != follower_answer:
                self._issue_warning(
                    "Follower failed to mirror attacker from previous round.",
                    {
                        "question_id": question.question_id,
                        "round": round_index + 1,
                        "attacker_round": round_index,
                        "attacker_answer": anchor,
                        "follower_answer": follower_answer,
                    },
                    question_state=question_state,
                )

    def _get_attacker_answer(
        self, history: List[AgentTurnRecord], round_index: int
    ) -> Optional[str]:
        if not self.attacker_id:
            return None
        for turn in history:
            if turn.round_index == round_index and turn.agent_id == self.attacker_id:
                if turn.parsed_answer:
                    return turn.parsed_answer.strip().upper()
        return None

    def _issue_warning(
        self,
        message: str,
        details: Dict[str, Any],
        question_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {"message": message, **details}
        self.role_warnings.append(message)
        if question_state is not None:
            question_state.setdefault("warnings", []).append(payload)
        self.logger.log_warning(payload)

    def _build_consensus(self, question: CommonsenseQuestion, history: List[AgentTurnRecord]) -> ConsensusRecord:
        last_round = max((turn.round_index for turn in history), default=0)
        final_turns = [turn for turn in history if turn.round_index == last_round]
        votes: Counter[str] = Counter()
        confidences: Dict[str, float] = {}
        for turn in final_turns:
            if not turn.parsed_answer:
                continue
            votes[turn.parsed_answer] += 1
            if turn.confidence is not None:
                confidences.setdefault(turn.parsed_answer, 0.0)
                confidences[turn.parsed_answer] += turn.confidence

        final_answer = None
        supporting_agents: List[str] = []
        confidence = None
        if votes:
            if self.config.debate.consensus_method == "confidence_weighted" and confidences:
                final_answer = max(confidences.items(), key=lambda item: item[1])[0]
            else:
                final_answer = votes.most_common(1)[0][0]
            supporting_agents = [turn.agent_id for turn in final_turns if turn.parsed_answer == final_answer]
            if confidences and final_answer in confidences:
                confidence = confidences[final_answer] / max(len(supporting_agents), 1)

        is_correct = None
        if question.answer_key and final_answer:
            is_correct = question.answer_key.strip().upper() == final_answer.strip().upper()

        return ConsensusRecord(
            question_id=question.question_id,
            method=self.config.debate.consensus_method,
            final_answer=final_answer,
            supporting_agents=supporting_agents,
            votes=dict(votes),
            confidence=confidence,
            is_correct=is_correct,
        )
