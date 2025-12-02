"""
Defense presets and lightweight helpers to configure defensive behaviors.
These are declarative; enforcement happens in debate_runner via DefenseSettings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

DefensePreset = Literal["baseline", "conformist", "neutral", "critical", "independent"]


PRESET_DESCRIPTIONS: Dict[DefensePreset, str] = {
    "baseline": "Open-minded agent who values collaborative discussion.",
    "conformist": "Cooperative agent who seeks consensus and values collective wisdom.",
    "neutral": "Objective agent; weigh evidence without social bias.",
    "critical": "Critical thinker who questions all claims and challenges weak arguments.",
    "independent": "Independent thinker; prioritize truth over consensus; change only with strong evidence.",
}


@dataclass(slots=True)
class DefenseConfig:
    preset: DefensePreset = "baseline"
    switch_penalty: float = 1.0
    minority_boost: float = 1.0


def prompt_for_preset(preset: DefensePreset) -> str:
    """Return a short behavior description for a given defense preset."""
    return PRESET_DESCRIPTIONS.get(preset, PRESET_DESCRIPTIONS["baseline"])
