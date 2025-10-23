from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from schema import BaseSignal, ValidationResult


class ClueValidator(ABC):
    """Domain-specific semantic checks for extracted signals."""

    @abstractmethod
    def validate_semantic(self, signal: BaseSignal) -> ValidationResult:
        """Validate a single signal according to plugin-specific rules."""


class NullValidator(ClueValidator):
    """Default validator that always passes semantic checks."""

    def validate_semantic(self, signal: BaseSignal) -> ValidationResult:  # noqa: D401
        return ValidationResult.ok(level="semantic")


class ClueExtractor(ABC):
    """Interface for pluggable clue extractors."""

    @property
    @abstractmethod
    def clue_id(self) -> str:
        """Unique identifier used for ids and registry lookup."""

    @abstractmethod
    def extract(self, scene_text: str, scene_id: int) -> Sequence[BaseSignal]:
        """Extract signals from a single scene."""

    def batch_extract(self, items: Iterable[tuple[int, str]]) -> Sequence[BaseSignal]:
        """Optional batch extraction hook; defaults to sequential extract calls."""

        outputs: list[BaseSignal] = []
        for scene_id, text in items:
            outputs.extend(self.extract(text, scene_id))
        return outputs

    def score(self, clue: BaseSignal) -> float:
        """Return relative importance for bundling/selection; defaults to 0."""

        _ = clue
        return 0.0

    def validator(self) -> ClueValidator:
        """Return semantic validator for this extractor."""

        return NullValidator()
