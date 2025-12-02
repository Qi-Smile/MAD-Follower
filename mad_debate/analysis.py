"""
Lightweight analysis helpers that wrap metrics into a single callable surface.
This keeps experiment scripts thin while providing reusable primitives for
ASR/CR, transition matrices, and token usage summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .metrics import (
    AnswerTrajectory,
    TransitionMatrix,
    build_answer_trajectories,
    compute_answer_change_histogram,
    compute_attack_success_rate,
    compute_conformity_rate,
    compute_drift,
    compute_transition_matrix,
    load_consensus_records,
    load_turn_logs,
    summarize_usage,
)


def load_all(log_path: Path) -> Tuple[List[AnswerTrajectory], TransitionMatrix, Dict[str, float]]:
    """Load turns and consensus, then derive trajectories, transitions, and usage."""
    turns = load_turn_logs(log_path)
    consensus = load_consensus_records(log_path)
    trajectories = build_answer_trajectories(turns)
    transition = compute_transition_matrix(trajectories, reference="ground_truth")
    usage = summarize_usage(turns)
    return trajectories, transition, usage


def summarize_core_metrics(
    log_path: Path,
    attacker_id: str = "attacker",
    follower_id: Optional[str] = "follower",
    exclude_agent_ids: Optional[Set[str]] = None,
) -> Dict[str, object]:
    turns = load_turn_logs(log_path)
    consensus = load_consensus_records(log_path)
    trajectories = build_answer_trajectories(turns)

    excluded = set(exclude_agent_ids or set())
    excluded.add(attacker_id)
    if follower_id:
        excluded.add(follower_id)

    drift = compute_drift(
        turns,
        attacker_id=attacker_id,
        follower_id=follower_id,
        exclude_agent_ids=excluded,
    )
    conformity = compute_conformity_rate(
        turns,
        attacker_id=attacker_id,
        follower_id=follower_id,
        reference="attacker",
        exclude_agent_ids=excluded,
    )
    attack_success = compute_attack_success_rate(trajectories, consensus)
    transition = compute_transition_matrix(
        trajectories,
        reference="ground_truth",
        exclude_agent_ids=excluded,
    )
    usage = summarize_usage(turns)
    change_hist = compute_answer_change_histogram(trajectories)

    return {
        "drift": drift,
        "conformity": conformity,
        "attack_success": attack_success,
        "transition": transition,
        "usage": usage,
        "change_hist": change_hist,
        "trajectories": trajectories,
        "consensus": consensus,
    }
