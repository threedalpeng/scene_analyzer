from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from clues.act import ActClue
from schema import AliasGroups, LLMAdjudication


@dataclass(slots=True)
class AliasingResult:
    alias_groups: AliasGroups
    alias_map: Dict[str, str]


@dataclass(slots=True)
class TemporalResult:
    fabula_rank: Dict[int, int]


@dataclass(slots=True)
class SynthesisResult:
    acts_representative: List[ActClue]
    acts_directed: List[ActClue]
    dyad_results: Dict[Tuple[str, str], LLMAdjudication]


@dataclass(slots=True)
class ValidationSummary:
    total: int
    passed: int
    failed: int
    warnings: List[str]


__all__ = [
    "AliasingResult",
    "TemporalResult",
    "SynthesisResult",
    "ValidationSummary",
]
