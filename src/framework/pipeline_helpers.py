from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Type

from framework.base import ClueExtractor
from framework.batch import CombinedBatchExtractor
from framework.checkpoint import (
    CheckpointManager,
    CheckpointState,
    compute_digest,
    deserialize_pipeline_result,
    serialize_pipeline_result,
    type_name,
)
from framework.result import PipelineResult
from schema import BaseClue


def prepare_checkpoint(
    config_checkpoint_enabled: bool,
    output_dir: Path | None,
    extractors: Sequence[ClueExtractor],
    processors: Sequence[Any],
) -> Tuple[
    CheckpointManager | None,
    list[str],
    list[str],
    dict[str, Type[BaseClue]],
]:
    extractor_keys = [extractor.checkpoint_id() for extractor in extractors]
    processor_keys = [type_name(proc.__class__) for proc in processors]

    clue_class_map: dict[str, Type[BaseClue]] = {}
    for extractor in extractors:
        for member in extractor.registry_members():
            clue_cls = member.clue_type
            clue_class_map[type_name(clue_cls)] = clue_cls

    checkpoint_manager: CheckpointManager | None = None
    if config_checkpoint_enabled:
        if output_dir is None:
            raise ValueError(
                "checkpoint_enabled requires an output_dir for checkpoint storage"
            )
        checkpoint_manager = CheckpointManager(output_dir / "checkpoint.json")

    return checkpoint_manager, extractor_keys, processor_keys, clue_class_map


def load_or_initialize(
    checkpoint_manager: CheckpointManager | None,
    resume: bool,
    scenes: Sequence[Mapping[str, Any]],
    metadata: Mapping[int, Mapping[str, Any]],
    extractor_keys: Sequence[str],
    processor_keys: Sequence[str],
    clue_class_map: Mapping[str, Type[BaseClue]],
    result: PipelineResult,
) -> Tuple[
    bool,
    dict[str, set[int]],
    int,
    list[Mapping[str, object]] | None,
]:
    extracted_state: dict[str, set[int]] = {}
    next_extractor_index = 0

    if checkpoint_manager is None or not resume:
        return False, extracted_state, next_extractor_index, None

    checkpoint_state = checkpoint_manager.load()
    if checkpoint_state is None:
        return False, extracted_state, next_extractor_index, None

    scenes_digest = compute_digest(scenes)
    metadata_digest = compute_digest(metadata)

    if not checkpoint_state.matches(
        scenes_digest,
        metadata_digest,
        extractor_keys,
        processor_keys,
    ):
        checkpoint_manager.clear()
        return False, extracted_state, next_extractor_index, None

    restored = deserialize_pipeline_result(
        checkpoint_state.result_payload, clue_class_map
    )
    result.metadata = restored.metadata
    result.scenes = restored.scenes
    result.context = restored.context
    result.participants = restored.participants
    for clue_type, clues in restored.iter_clue_items():
        result.put_clues(clue_type, clues)

    extracted_state = {
        key: set(value) for key, value in checkpoint_state.extractor_completed.items()
    }
    next_extractor_index = checkpoint_state.next_extractor_index
    processor_payloads = list(checkpoint_state.processor_payloads or [])
    return True, extracted_state, next_extractor_index, processor_payloads


def save_checkpoint(
    checkpoint_manager: CheckpointManager | None,
    dirty: bool,
    *,
    force: bool = False,
    scenes: Sequence[Mapping[str, Any]],
    metadata: Mapping[int, Mapping[str, Any]],
    extractor_keys: Sequence[str],
    processor_keys: Sequence[str],
    clue_class_map: Mapping[str, Type[BaseClue]],
    extractor_completed: Mapping[str, Iterable[int]],
    result: PipelineResult,
    next_extractor_index: int,
    next_processor_index: int,
    processor_payloads: Sequence[Mapping[str, object]] | None,
) -> None:
    if checkpoint_manager is None or (not dirty and not force):
        return

    clue_name_map = {cls: name for name, cls in clue_class_map.items()}

    state = CheckpointState(
        scenes_digest=compute_digest(scenes),
        metadata_digest=compute_digest(metadata),
        extractor_order=list(extractor_keys),
        processor_order=list(processor_keys),
        next_extractor_index=next_extractor_index,
        next_processor_index=next_processor_index,
        extractor_completed={
            key: sorted({int(sid) for sid in values})
            for key, values in extractor_completed.items()
        },
        result_payload=serialize_pipeline_result(result, clue_name_map),
        processor_payloads=list(processor_payloads or []),
    )
    checkpoint_manager.save(state)


def merge_extractor_output(
    extractor: ClueExtractor,
    clues: Sequence[BaseClue],
    result: PipelineResult,
) -> None:
    if not clues:
        return

    if isinstance(extractor, CombinedBatchExtractor):
        buckets: Dict[Type[BaseClue], list[BaseClue]] = defaultdict(list)
        for clue in clues:
            buckets[type(clue)].append(clue)
        for clue_type, values in buckets.items():
            result.append_clues(clue_type, values)
    else:
        result.append_clues(extractor.clue_type, clues)
