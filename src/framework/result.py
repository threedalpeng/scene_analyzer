from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Type,
    TypeVar,
    overload,
)

from pydantic import BaseModel

from schema import BaseClue, ValidationResult

T = TypeVar("T")
BaseClueT = TypeVar("BaseClueT", bound=BaseClue)


class PipelineResult:
    """Container for extracted clues, processor outputs, and validation reports."""

    def __init__(self) -> None:
        self._clues: Dict[Type[BaseClue], list[BaseClue]] = {}
        self._outputs: Dict[Type[BaseModel], BaseModel] = {}
        self.validation: list[tuple[BaseClue, Sequence[ValidationResult]]] = []
        self.metadata: Dict[int, dict] = {}
        self.scenes: list[dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self.participants: Dict[int, list[str]] = {}

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

    def put_output(self, output: BaseModel) -> BaseModel:
        if not isinstance(output, BaseModel):
            raise TypeError(
                f"Processor output must be a BaseModel, got {type(output).__name__}"
            )
        self._outputs[type(output)] = output
        return output

    def put_participants(self, scene_id: int, names: Iterable[str]) -> None:
        normalized = []
        for name in names:
            name = name.strip()
            if not name:
                continue
            if name not in normalized:
                normalized.append(name)
        if normalized:
            self.participants[int(scene_id)] = normalized

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


__all__ = ["PipelineResult"]
