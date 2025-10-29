from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    Type,
    TypeVar,
    cast,
)

from pydantic import BaseModel

from framework.schema import BaseClue, ValidationResult

TOutput = TypeVar("TOutput", bound=BaseModel)
BaseClueT = TypeVar("BaseClueT", bound=BaseClue)


@dataclass(slots=True)
class ExecutionFailure:
    """Record of a failure encountered during pipeline execution."""

    segment: int
    stage: str
    component: str
    error: str
    timestamp: str


class PipelineResult:
    """
    Aggregate view of artifacts produced during a pipeline run.

    A :class:`PipelineResult` stores the normalized input segments alongside
    extracted clues, processor outputs, validation reports, execution failures,
    and arbitrary context shared between stages. Collections returned by the
    public getters are copies so downstream code can mutate them safely.

    Examples:
        >>> acts = result.get_clues(ActClue)
        >>> synthesis = result.get_output(SynthesisResult)
        >>> if result.failures:
        ...     for failure in result.failures:
        ...         print(failure.component, failure.error)
    """

    def __init__(
        self,
        *,
        segments: Sequence[Mapping[str, Any]] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self._clues: dict[Type[BaseClue], list[BaseClue]] = {}
        self._outputs: dict[Type[BaseModel], BaseModel] = {}
        self.validation: list[tuple[BaseClue, Sequence[ValidationResult]]] = []
        self.segments: list[dict[str, Any]] = [dict(item) for item in (segments or [])]
        self.context: dict[str, Any] = {}
        self.failures: list[ExecutionFailure] = []
        if context:
            self.context.update(dict(context))

    def get_clues(self, clue_type: Type[BaseClueT]) -> list[BaseClueT]:
        """
        Return a copy of the clues stored for the given type.

        The returned list is detached from internal storage so callers may
        mutate it without affecting the backing collection.
        """

        if clue_type is BaseClue:
            return cast(list[BaseClueT], list(self.all_clues))
        bucket = self._clues.get(clue_type, [])
        return cast(list[BaseClueT], list(bucket))

    def set_clues(self, clue_type: Type[BaseClue], clues: Iterable[BaseClue]) -> None:
        """Replace all stored clues for ``clue_type`` with ``clues``."""

        self._clues[clue_type] = list(clues)

    def add_clues(self, clue_type: Type[BaseClue], clues: Iterable[BaseClue]) -> None:
        """Append ``clues`` to the collection stored for ``clue_type``."""

        bucket = self._clues.setdefault(clue_type, [])
        bucket.extend(clues)

    def get_output(self, output_type: Type[TOutput]) -> TOutput | None:
        """
        Return the processor output matching ``output_type`` if available.

        Outputs are stored by their concrete :class:`pydantic.BaseModel` type.
        """

        value = self._outputs.get(output_type)
        if value is not None:
            assert isinstance(value, output_type)
        return value

    @property
    def all_clues(self) -> list[BaseClue]:
        out: list[BaseClue] = []
        for clues in self._clues.values():
            out.extend(clues)
        return out

    def put_output(self, output: TOutput) -> TOutput:
        """Store a processor output, keyed by its concrete model type."""
        if not isinstance(output, BaseModel):
            raise TypeError(
                f"Processor output must be a BaseModel, got {type(output).__name__}"
            )
        self._outputs[type(output)] = output
        return output

    def record_failure(
        self,
        segment: int,
        stage: str,
        component: str,
        error: Exception | str,
    ) -> None:
        """
        Record a failure encountered while executing the pipeline.

        Args:
            segment: Segment identifier associated with the failure, or ``0``
                when the error is not segment-specific.
            stage: Execution stage name (e.g., ``"extraction"``).
            component: Extractor or processor responsible for the failure.
            error: Exception or message describing the failure.
        """

        self.failures.append(
            ExecutionFailure(
                segment=int(segment),
                stage=str(stage),
                component=str(component),
                error=str(error),
                timestamp=datetime.now().isoformat(),
            )
        )

    @property
    def clue_index(self) -> Mapping[Type[BaseClue], Sequence[BaseClue]]:
        return self._clues

    def iter_clue_items(self) -> Iterator[tuple[Type[BaseClue], list[BaseClue]]]:
        for clue_type, clues in self._clues.items():
            yield clue_type, list(clues)

    def iter_outputs(self) -> Iterator[tuple[Type[BaseModel], BaseModel]]:
        for output_type, value in self._outputs.items():
            yield output_type, value

    def clear_outputs(self) -> None:
        self._outputs.clear()

    def merge_context(self, values: Mapping[str, Any]) -> None:
        """Merge ``values`` into the shared context dictionary."""
        for key, value in values.items():
            self.context[key] = value

    def get_segment(self, segment_id: int) -> dict[str, Any] | None:
        for segment in self.segments:
            if int(segment.get("segment", -1)) == int(segment_id):
                return segment
        return None

    def get_segments_by_level(self, level: str) -> list[dict[str, Any]]:
        return [seg for seg in self.segments if seg.get("level") == level]

    def get_children(self, parent_id: int) -> list[dict[str, Any]]:
        return [seg for seg in self.segments if seg.get("parent") == parent_id]

    def get_hierarchy(self) -> dict[str, list[int]]:
        from collections import defaultdict

        hierarchy: dict[str, list[int]] = defaultdict(list)
        for seg in self.segments:
            level = seg.get("level", "default")
            try:
                seg_id = int(seg["segment"])
            except (KeyError, TypeError, ValueError):
                continue
            hierarchy[level].append(seg_id)
        return dict(hierarchy)

    def set_context(
        self,
        key: str,
        value: Any,
        *,
        namespace: Literal["framework", "processor", "user"] = "user",
    ) -> None:
        full_key = f"{namespace}.{key}" if "." not in key else key
        self.context[full_key] = value

    def get_context(
        self,
        key: str,
        *,
        namespace: Literal["framework", "processor", "user"] = "user",
        default: Any | None = None,
    ) -> Any | None:
        """Retrieve a namespaced context value."""
        full_key = f"{namespace}.{key}" if "." not in key else key
        return self.context.get(full_key, default)

    def save(self, path: Path) -> None:
        import pickle

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "PipelineResult":
        import pickle

        target = Path(path)
        return pickle.loads(target.read_bytes())

    def copy(self) -> "PipelineResult":
        import copy

        return copy.deepcopy(self)


__all__ = ["PipelineResult", "ExecutionFailure"]
