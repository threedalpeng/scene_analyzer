from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Mapping,
    Sequence,
    Type,
    TypeVar,
)

from framework.schema import BaseClue, ValidationResult

if TYPE_CHECKING:
    from framework.core.pipeline import PipelineConfig


ClueT = TypeVar("ClueT", bound=BaseClue)


class ClueValidator(ABC):
    """
    Domain validator returned by :class:`ClueExtractor` implementations.

    Subclasses implement semantic and (optionally) coherence checks that run
    after structural validation has passed.
    """

    @abstractmethod
    def validate_semantic(self, clue: BaseClue) -> ValidationResult:
        """Validate domain-specific rules for a single clue."""

    def validate_coherence(
        self, clue: BaseClue, context: Mapping[str, Any] | None = None
    ) -> ValidationResult | None:
        """
        Optionally validate cross-clue coherence.

        Return ``None`` to skip if no coherence logic is required.
        """

        _ = clue, context
        return None


class ClueExtractor(Generic[ClueT], ABC):
    """
    Base class for pluggable clue extractors.

    Extractors transform raw segment text into structured clue objects. The
    default implementation supports either single-segment extraction via
    :meth:`extract` or batch extraction via :meth:`batch_extract`.
    """

    def __init__(self) -> None:
        self._configured = False
        self._failure_recorder: Callable[[int, str], None] | None = None

    def configure(self, config: "PipelineConfig") -> None:
        """
        Inject shared pipeline configuration before extraction begins.

        Subclasses should override this method to capture values from the
        pipeline configuration (for example, LLM clients or batch sizes) and
        must call ``super().configure(config)`` if the default bookkeeping is
        required.
        """

        self._configured = True

    def set_failure_recorder(
        self, recorder: Callable[[int, str], None] | None
    ) -> None:
        """Attach a callback that records per-segment extraction failures."""

        self._failure_recorder = recorder

    def _record_failure(self, segment_id: int, error: Exception | str) -> None:
        if self._failure_recorder:
            self._failure_recorder(int(segment_id), str(error))

    @property
    @abstractmethod
    def clue_type(self) -> Type[ClueT]:
        """Concrete BaseClue subtype produced by this extractor."""

    @abstractmethod
    def extract(self, segment_text: str, segment_id: int) -> Sequence[ClueT]:
        """Extract clues from a single segment."""

    def batch_extract(self, items: Iterable[tuple[int, str]]) -> Sequence[ClueT]:
        """
        Optional batch extraction hook.

        The default implementation falls back to sequential :meth:`extract`
        calls for each ``(segment_id, text)`` pair supplied.
        """

        outputs: list[ClueT] = []
        for segment_id, text in items:
            outputs.extend(self.extract(text, segment_id))
        return outputs

    def score(self, clue: ClueT) -> float:
        """Relative importance used for downstream ranking; defaults to ``0``."""

        _ = clue
        return 0.0

    def validator(self) -> ClueValidator:
        """Return the semantic/coherence validator for this extractor."""

        return NullValidator()

    def participants(self) -> Mapping[int, list[str]]:
        """
        Return detected participant names keyed by segment id.

        The default implementation returns an empty mapping.
        """
        return {}

    def registry_members(self) -> Sequence["ClueExtractor[Any]"]:
        """
        Return extractors that should be registered for validation.

        Override when a batch extractor wraps multiple extractor instances that
        expose their own validators.
        """
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
