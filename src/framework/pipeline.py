from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Type

from google import genai

from framework.base import ClueExtractor
from framework.batch import BatchExtractor, CombinedBatchExtractor
from framework.pipeline_helpers import (
    load_or_initialize,
    merge_extractor_output,
    prepare_checkpoint,
    save_checkpoint,
)
from framework.processor import Processor
from framework.registry import ClueRegistry
from framework.result import PipelineResult
from framework.validation import ValidationContext, ValidationPipeline
from schema import BaseClue, ValidationResult


@dataclass(slots=True)
class PipelineConfig:
    client: genai.Client | None = None
    batch_size: int = 50
    validate: bool = True
    strict_validation: bool = False
    checkpoint_enabled: bool = False
    output_dir: Path | None = None


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._extractors: list[ClueExtractor] = []
        self._processors: list[Processor] = []

    ##################
    ### Public API ###
    ##################

    def extract(
        self,
        extractor: Type[ClueExtractor]
        | ClueExtractor
        | list[Type[ClueExtractor] | ClueExtractor],
        /,
        **params: Any,
    ) -> "Pipeline":
        if isinstance(extractor, list):
            if params:
                raise ValueError(
                    "keyword overrides are not supported for extractor lists"
                )
            members = [self._instantiate_extractor(item, {}) for item in extractor]
            batch_members: list[BatchExtractor[Any]] = []
            for member in members:
                if not isinstance(member, BatchExtractor):
                    raise TypeError(
                        "Combined extraction requires BatchExtractor instances; "
                        f"got {type(member).__name__}"
                    )
                batch_members.append(member)
            combined = CombinedBatchExtractor(batch_members)
            self._extractors.append(combined)
        else:
            instance = self._instantiate_extractor(extractor, params)
            self._extractors.append(instance)
        return self

    def process(self, processor: Processor) -> "Pipeline":
        self._processors.append(processor)
        return self

    def run(
        self,
        scenes: list[dict],
        *,
        metadata: dict[int, dict] | None = None,
        context: Mapping[str, Any] | None = None,
        resume: bool = False,
    ) -> PipelineResult:
        metadata = metadata or {}
        self._configure_extractors()
        self._configure_processors()

        scene_pairs: list[tuple[int, str]] = [
            (int(item["scene"]), str(item["text"])) for item in scenes
        ]

        if not scene_pairs:
            raise ValueError("Pipeline requires at least one scene")

        (
            checkpoint_manager,
            extractor_keys,
            processor_keys,
            clue_class_map,
        ) = prepare_checkpoint(
            self.config.checkpoint_enabled,
            self.config.output_dir,
            self._extractors,
            self._processors,
        )

        result = PipelineResult()
        (
            resumed,
            extractor_completed,
            next_extractor_index,
            processor_payloads,
        ) = load_or_initialize(
            checkpoint_manager,
            resume,
            scenes,
            metadata,
            extractor_keys,
            processor_keys,
            clue_class_map,
            result,
        )

        if not resumed:
            result.metadata = {int(k): dict(v) for k, v in metadata.items()}
            result.scenes = [dict(item) for item in scenes]
            if context:
                result.context.update(dict(context))

        processor_payloads_list: list[Mapping[str, object] | None] = [
            payload if isinstance(payload, Mapping) else None
            for payload in (processor_payloads or [])
        ]

        next_processor_index = 0

        while len(processor_payloads_list) < len(self._processors):
            processor_payloads_list.append(None)

        registry = ClueRegistry()
        for extractor in self._extractors:
            for member in extractor.registry_members():
                registry.register(member)

        def save_state(dirty: bool, *, force: bool = False) -> None:
            save_checkpoint(
                checkpoint_manager,
                dirty,
                force=force,
                scenes=scenes,
                metadata=metadata,
                extractor_keys=extractor_keys,
                processor_keys=processor_keys,
                clue_class_map=clue_class_map,
                extractor_completed=extractor_completed,
                result=result,
                next_extractor_index=next_extractor_index,
                next_processor_index=next_processor_index,
                processor_payloads=[
                    payload
                    for payload in processor_payloads_list
                    if payload is not None
                ],
            )

        for idx, extractor in enumerate(self._extractors):
            key = extractor.checkpoint_id()
            completed = extractor_completed.setdefault(key, set())

            if idx < next_extractor_index:
                continue

            if isinstance(extractor, BatchExtractor):
                extractor.load_checkpoint(completed)
                batch_size = max(1, extractor.effective_batch_size())
            else:
                batch_size = len(scene_pairs)

            remaining_pairs = [pair for pair in scene_pairs if pair[0] not in completed]

            if not remaining_pairs:
                next_extractor_index = idx + 1
                save_state(dirty=False)
                continue

            for chunk in _chunk_pairs(remaining_pairs, batch_size):
                if not chunk:
                    continue
                chunk_clues = list(extractor.batch_extract(chunk))

                merge_extractor_output(extractor, chunk_clues, result)

                participants_map = extractor.participants()
                processed_ids: list[int] = []
                for scene_id, _ in chunk:
                    names = participants_map.get(scene_id)
                    if names:
                        result.put_participants(scene_id, names)
                        processed_ids.append(scene_id)

                if processed_ids:
                    completed.update(processed_ids)
                    extractor_completed[key] = completed
                    next_extractor_index = idx
                    save_state(dirty=True)

            next_extractor_index = idx + 1
            save_state(dirty=True)

        if self.config.validate:
            validator = ValidationPipeline(registry)
            validation_context = ValidationContext(
                known_scenes={sid for sid, _ in scene_pairs},
            )
            validations = validator.validate_batch(
                result.all_clues, context=validation_context
            )
            result.validation = validations
            if self.config.strict_validation:
                self._raise_on_validation_errors(validations)

        start_processor_index = 0
        for idx, processor in enumerate(self._processors):
            payload = processor_payloads_list[idx]
            if payload is None:
                start_processor_index = idx
                break
            restored = processor.restore_from_checkpoint(payload, result)
            if restored is not None:
                result.put_output(restored)
            start_processor_index = idx + 1

        next_processor_index = start_processor_index

        for idx in range(start_processor_index, len(self._processors)):
            processor = self._processors[idx]
            output = processor(result)
            if output is not None:
                result.put_output(output)
            processor_payloads_list[idx] = processor.checkpoint_state(result, output)
            next_processor_index = idx + 1
            save_state(dirty=True)

        if checkpoint_manager is not None:
            checkpoint_manager.clear()

        return result

    def _configure_extractors(self) -> None:
        for extractor in self._extractors:
            extractor.configure(self.config)

    def _configure_processors(self) -> None:
        for processor in self._processors:
            processor.configure(self.config)

    def _instantiate_extractor(
        self,
        extractor: Type[ClueExtractor] | ClueExtractor,
        params: Mapping[str, Any],
    ) -> ClueExtractor:
        if isinstance(extractor, type):
            return extractor(**params)
        if params:
            raise ValueError("keyword overrides require extractor class input")
        return extractor

    @staticmethod
    def _raise_on_validation_errors(
        validations: Sequence[tuple[BaseClue, Sequence[ValidationResult]]],
    ) -> None:
        errors: list[str] = []
        for clue, records in validations:
            for record in records:
                if not record.passed:
                    errors.append(
                        f"Scene {clue.scene} | {clue.id} ({clue.clue_type}) "
                        f"failed {record.level}: {record.errors}"
                    )

        if errors:
            summary = f"Validation failed: {len(errors)} error(s) found\n"
            raise ValueError(summary + "\n".join(errors))


def _chunk_pairs(
    pairs: Sequence[tuple[int, str]], batch_size: int
) -> Iterable[list[tuple[int, str]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive for checkpointed execution")
    for i in range(0, len(pairs), batch_size):
        yield list(pairs[i : i + batch_size])
