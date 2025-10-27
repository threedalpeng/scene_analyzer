from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple, Type

from framework.result import PipelineResult
from schema import BaseClue

Serializer = Callable[[Any], Dict[str, Any]]
Deserializer = Callable[[Mapping[str, Any]], Any]
CodecEntry = Tuple[Type[Any], Serializer, Deserializer]


def type_name(cls: Type[Any]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def compute_digest(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def serialize_pipeline_result(
    result: PipelineResult,
    clue_name_map: Mapping[Type[BaseClue], str],
) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "metadata": result.metadata,
        "scenes": result.scenes,
        "context": result.context,
        "participants": result.participants,
        "clues": {},
    }

    for clue_type, clues in result.iter_clue_items():
        key = clue_name_map.get(clue_type)
        if key is None:
            continue
        data["clues"][key] = [clue.model_dump() for clue in clues]

    return data


def deserialize_pipeline_result(
    payload: Mapping[str, Any],
    clue_class_map: Mapping[str, Type[BaseClue]],
) -> PipelineResult:
    result = PipelineResult()
    result.metadata = {int(k): dict(v) for k, v in payload["metadata"].items()}
    result.scenes = [dict(item) for item in payload["scenes"]]
    result.context = dict(payload.get("context", {}))

    participants: Dict[int, list[str]] = {}
    for sid, names in payload.get("participants", {}).items():
        participants[int(sid)] = list(names)
    result.participants = participants

    for clue_name, items in payload.get("clues", {}).items():
        clue_cls = clue_class_map.get(clue_name)
        if clue_cls is None:
            continue
        clues = [clue_cls.model_validate(item) for item in items]
        result.put_clues(clue_cls, clues)

    return result


@dataclass
class CheckpointState:
    version: int = 1
    scenes_digest: str = ""
    metadata_digest: str = ""
    extractor_order: Sequence[str] = field(default_factory=list)
    processor_order: Sequence[str] = field(default_factory=list)
    next_extractor_index: int = 0
    next_processor_index: int = 0
    extractor_completed: Dict[str, Sequence[int]] = field(default_factory=dict)
    result_payload: Dict[str, Any] = field(default_factory=dict)
    processor_payloads: List[Mapping[str, object]] | None = None

    def matches(
        self,
        scenes_digest: str,
        metadata_digest: str,
        extractor_order: Sequence[str],
        processor_order: Sequence[str],
    ) -> bool:
        return (
            self.version == 1
            and self.scenes_digest == scenes_digest
            and self.metadata_digest == metadata_digest
            and list(self.extractor_order) == list(extractor_order)
            and list(self.processor_order) == list(processor_order)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "scenes_digest": self.scenes_digest,
            "metadata_digest": self.metadata_digest,
            "extractor_order": list(self.extractor_order),
            "processor_order": list(self.processor_order),
            "next_extractor_index": self.next_extractor_index,
            "next_processor_index": self.next_processor_index,
            "extractor_completed": {
                key: list(value) for key, value in self.extractor_completed.items()
            },
            "result": self.result_payload,
            "processor_payloads": list(self.processor_payloads or []),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointState":
        return cls(
            version=int(payload.get("version", 0)),
            scenes_digest=str(payload.get("scenes_digest", "")),
            metadata_digest=str(payload.get("metadata_digest", "")),
            extractor_order=list(payload.get("extractor_order", [])),
            processor_order=list(payload.get("processor_order", [])),
            next_extractor_index=int(payload.get("next_extractor_index", 0)),
            next_processor_index=int(payload.get("next_processor_index", 0)),
            extractor_completed={
                key: [int(sid) for sid in value]
                for key, value in payload.get("extractor_completed", {}).items()
            },
            result_payload=dict(payload.get("result", {})),
            processor_payloads=list(payload.get("processor_payloads", []) or []),
        )


class CheckpointManager:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> CheckpointState | None:
        if not self.path.exists():
            return None
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return CheckpointState.from_dict(raw)

    def save(self, state: CheckpointState) -> None:
        _ensure_parent(self.path)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.path)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


__all__ = [
    "CheckpointManager",
    "CheckpointState",
    "compute_digest",
    "deserialize_pipeline_result",
    "serialize_pipeline_result",
    "type_name",
]
