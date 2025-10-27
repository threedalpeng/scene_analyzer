from __future__ import annotations

from typing import (
    TYPE_CHECKING,
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

if TYPE_CHECKING:
    from framework.registry import ClueRegistry, ProcessorResultRegistry

T = TypeVar("T")
BaseClueT = TypeVar("BaseClueT", bound=BaseClue)


class PipelineResultSnapshot(BaseModel):
    """Serializable view of PipelineResult for checkpointing."""

    scenes: list[dict[str, Any]]
    metadata: dict[int, dict[str, Any]]
    context: dict[str, Any]
    participants: dict[int, list[str]]
    clues: dict[str, list[dict[str, Any]]]
    outputs: dict[str, dict[str, Any]]


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

    def checkpoint_state(self) -> PipelineResultSnapshot:
        """Materialize the current state into a checkpoint snapshot."""

        return PipelineResultSnapshot(
            scenes=[dict(scene) for scene in self.scenes],
            metadata={int(k): dict(v) for k, v in self.metadata.items()},
            context=dict(self.context),
            participants={int(k): list(v) for k, v in self.participants.items()},
            clues={
                clue_type.__name__: [clue.model_dump() for clue in clues]
                for clue_type, clues in self._clues.items()
            },
            outputs={
                type(output).__name__: output.model_dump()
                for output in self._outputs.values()
            },
        )

    def restore_state(
        self,
        snapshot: PipelineResultSnapshot,
        registry: "ClueRegistry",
        result_registry: "ProcessorResultRegistry",
    ) -> None:
        """Restore state from a previously captured snapshot."""

        self.scenes = [dict(scene) for scene in snapshot.scenes]
        self.metadata = {int(k): dict(v) for k, v in snapshot.metadata.items()}
        self.context = dict(snapshot.context)
        self.participants = {
            int(k): list(v) for k, v in snapshot.participants.items()
        }

        self._clues.clear()
        for clue_name, clue_dicts in snapshot.clues.items():
            clue_type = registry.get_type(clue_name)
            self._clues[clue_type] = [
                clue_type.model_validate(clue_dict) for clue_dict in clue_dicts
            ]

        self._outputs.clear()
        for output_name, data in snapshot.outputs.items():
            if result_registry.has_type(output_name):
                output_type = result_registry.get_type(output_name)
                self._outputs[output_type] = output_type.model_validate(data)


__all__ = ["PipelineResult", "PipelineResultSnapshot"]
