from __future__ import annotations

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
    overload,
)

from pydantic import BaseModel, Field

from framework.schema import BaseClue, ValidationResult

T = TypeVar("T")
TOutput = TypeVar("TOutput", bound=BaseModel)
BaseClueT = TypeVar("BaseClueT", bound=BaseClue)


class ParticipantsOutput(BaseModel):
    by_segment: dict[int, list[str]] = Field(default_factory=dict)

    def add(self, segment_id: int, names: Iterable[str]) -> None:
        bucket = self.by_segment.setdefault(int(segment_id), [])
        for name in names:
            normalized = name.strip()
            if not normalized:
                continue
            if normalized not in bucket:
                bucket.append(normalized)


class PipelineResult:
    """Container for extracted clues, processor outputs, and validation reports."""

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
        if context:
            self.context.update(dict(context))

    @overload
    def get(self, t: Type[BaseClueT]) -> list[BaseClueT]: ...

    @overload
    def get(self, t: Type[T]) -> T | None: ...

    def get(self, t: Type[Any]) -> Any:
        """Type-based lookup for clues and processor outputs."""

        if isinstance(t, type) and issubclass(t, BaseClue):
            if t is BaseClue:
                return list(self.all_clues)
            bucket = self._clues.get(t, [])
            assert bucket is not None
            return list(bucket)
        value = self._outputs.get(t)
        if value is not None:
            assert isinstance(value, t)
        return value

    @property
    def all_clues(self) -> list[BaseClue]:
        out: list[BaseClue] = []
        for clues in self._clues.values():
            out.extend(clues)
        return out

    def put_clues(self, clue_type: Type[BaseClue], clues: Iterable[BaseClue]) -> None:
        self._clues[clue_type] = list(clues)

    def append_clues(
        self, clue_type: Type[BaseClue], clues: Iterable[BaseClue]
    ) -> None:
        bucket = self._clues.setdefault(clue_type, [])
        bucket.extend(clues)

    def put_output(self, output: TOutput) -> TOutput:
        if not isinstance(output, BaseModel):
            raise TypeError(
                f"Processor output must be a BaseModel, got {type(output).__name__}"
            )
        self._outputs[type(output)] = output
        return output

    def clues_for(self, clue_type: Type[BaseClue]) -> list[BaseClue]:
        return list(self._clues.get(clue_type, []))

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

    def add_participants(self, segment_id: int, names: Iterable[str]) -> None:
        participants = self.get(ParticipantsOutput)
        if participants is None:
            participants = self.put_output(ParticipantsOutput())
        participants.add(segment_id, names)

    def merge_context(self, values: Mapping[str, Any]) -> None:
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


__all__ = ["PipelineResult", "ParticipantsOutput"]
