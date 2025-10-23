from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Sequence

from schema import BaseClue, ValidationResult


class ClueValidator(ABC):
    """Domain-specific semantic checks for extracted clues."""

    @abstractmethod
    def validate_semantic(self, clue: BaseClue) -> ValidationResult:
        """Validate a single clue according to plugin-specific rules."""

    def validate_coherence(
        self, clue: BaseClue, context: Mapping[str, Any] | None = None
    ) -> ValidationResult | None:
        """Validate cross-clue coherence; return None to skip."""
        _ = clue, context
        return None


class NullValidator(ClueValidator):
    """Default validator that always passes semantic checks."""

    def validate_semantic(self, clue: BaseClue) -> ValidationResult:  # noqa: D401
        return ValidationResult.ok(level="semantic")

    def validate_coherence(
        self, clue: BaseClue, context: Mapping[str, Any] | None = None
    ) -> ValidationResult | None:  # noqa: D401
        _ = clue, context
        return None


class ClueExtractor(ABC):
    """Interface for pluggable clue extractors."""

    @property
    @abstractmethod
    def clue_type(self) -> str:
        """Clue type identifier used for ids and registry lookup."""

    @abstractmethod
    def extract(self, scene_text: str, scene_id: int) -> Sequence[BaseClue]:
        """Extract clues from a single scene."""

    def batch_extract(self, items: Iterable[tuple[int, str]]) -> Sequence[BaseClue]:
        """Optional batch extraction hook; defaults to sequential extract calls."""

        outputs: list[BaseClue] = []
        for scene_id, text in items:
            outputs.extend(self.extract(text, scene_id))
        return outputs

    def score(self, clue: BaseClue) -> float:
        """Return relative importance for bundling/selection; defaults to 0."""

        _ = clue
        return 0.0

    def validator(self) -> ClueValidator:
        """Return semantic validator for this extractor."""

        return NullValidator()
