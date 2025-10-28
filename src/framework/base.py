from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Mapping,
    Sequence,
    Type,
    TypeVar,
)
from schema import BaseClue, ValidationResult

if TYPE_CHECKING:
    from framework.pipeline import PipelineConfig


ClueT = TypeVar("ClueT", bound=BaseClue)


class ClueValidator(ABC):
    """Domain-specific validation hooks for a clue extractor."""

    @abstractmethod
    def validate_semantic(self, clue: BaseClue) -> ValidationResult:
        """Validate semantic rules for a single clue."""

    def validate_coherence(
        self, clue: BaseClue, context: Mapping[str, Any] | None = None
    ) -> ValidationResult | None:
        """Optional cross-clue validation; return None to skip."""

        _ = clue, context
        return None


class ClueExtractor(Generic[ClueT], ABC):
    """Interface for pluggable clue extractors."""

    def __init__(self) -> None:
        self._configured = False

    def configure(self, config: "PipelineConfig") -> None:
        """Inject shared pipeline configuration before extraction."""

        self._configured = True

    @property
    @abstractmethod
    def clue_type(self) -> Type[ClueT]:
        """Concrete BaseClue subtype produced by this extractor."""

    @abstractmethod
    def extract(self, scene_text: str, scene_id: int) -> Sequence[ClueT]:
        """Extract clues from a single scene."""

    def batch_extract(self, items: Iterable[tuple[int, str]]) -> Sequence[ClueT]:
        """Optional batch extraction hook; defaults to sequential extract calls."""

        outputs: list[ClueT] = []
        for scene_id, text in items:
            outputs.extend(self.extract(text, scene_id))
        return outputs

    def score(self, clue: ClueT) -> float:
        """Relative importance for bundling/selection; defaults to 0."""

        _ = clue
        return 0.0

    def validator(self) -> ClueValidator:
        """Return semantic validator for this extractor."""

        return NullValidator()

    def participants(self) -> Mapping[int, list[str]]:
        """Return participants by scene; defaults to empty dict."""
        return {}

    def registry_members(self) -> Sequence["ClueExtractor[Any]"]:
        """Return extractors that should be registered for validation."""
        return [self]

    def checkpoint_id(self) -> str:
        """Stable identifier used for checkpoint bookkeeping."""
        cls = self.__class__
        return f"{cls.__module__}.{cls.__qualname__}"


class NullValidator(ClueValidator):
    """Default validator that always passes semantic/coherence checks."""

    def validate_semantic(self, clue: BaseClue) -> ValidationResult:  # noqa: D401
        _ = clue
        return ValidationResult.ok(level="semantic")

    def validate_coherence(
        self, clue: BaseClue, context: Mapping[str, Any] | None = None
    ) -> ValidationResult | None:  # noqa: D401
        _ = clue, context
        return None


__all__ = [
    "ClueExtractor",
    "ClueValidator",
    "NullValidator",
]
