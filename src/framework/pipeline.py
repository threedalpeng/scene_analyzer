from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, Type, TypeGuard

from google import genai

from framework.base import ClueExtractor
from framework.batch import BatchExtractor, CombinedBatchExtractor
from framework.processor import Processor
from framework.registry import ClueRegistry
from framework.result import PipelineResult
from framework.validation import ValidationContext, ValidationPipeline
from schema import BaseClue


@dataclass(slots=True)
class PipelineConfig:
    client: genai.Client | None = None
    batch_size: int = 50
    validate: bool = True
    strict_validation: bool = False


CheckpointMarker = tuple[Literal["checkpoint"], str]


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._stages: list[list[ClueExtractor] | Processor | CheckpointMarker] = []
        self._registry = ClueRegistry()

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
            instance: ClueExtractor = CombinedBatchExtractor(batch_members)
        else:
            instance = self._instantiate_extractor(extractor, params)

        stage_members = [instance]
        self._register_extractor_stage(stage_members)
        self._stages.append(stage_members)
        return self

    def process(self, processor: Processor) -> "Pipeline":
        self._stages.append(processor)
        return self

    def pipe(self, other: "Pipeline") -> "Pipeline":
        for stage in other._stages:
            if isinstance(stage, list):
                new_stage = list(stage)
                self._register_extractor_stage(new_stage)
                self._stages.append(new_stage)
            elif self._is_processor_stage(stage):
                self._stages.append(stage)
            elif isinstance(stage, tuple) and stage[0] == "checkpoint":
                self._stages.append(stage)
            else:
                raise TypeError(
                    f"Unsupported pipeline stage type: {type(stage).__name__}"
                )

        return self

    def checkpoint(self, name: str) -> "Pipeline":
        if not name:
            raise ValueError("Checkpoint name must be a non-empty string")
        self._stages.append(("checkpoint", name))
        return self

    def run(
        self,
        scenes: list[dict],
        *,
        metadata: dict[int, dict] | None = None,
        context: Mapping[str, Any] | None = None,
        checkpoint_dir: Path | None = None,
        load_checkpoint: Path | None = None,
        resume_from: str | None = None,
        stop_at: str | None = None,
        auto_save: bool = False,
    ) -> PipelineResult:
        if not scenes and load_checkpoint is None:
            raise ValueError("Pipeline requires at least one scene")

        metadata = metadata or {}
        result, derived_resume = self._initialize_result(
            scenes, metadata, context, load_checkpoint
        )
        effective_resume = resume_from or derived_resume

        skip_until = effective_resume
        for stage in self._stages:
            if isinstance(stage, tuple) and stage[0] == "checkpoint":
                checkpoint_name = stage[1]
                if auto_save and checkpoint_dir is not None:
                    path = checkpoint_dir / f"{checkpoint_name}.pkl"
                    result.save(path)
                if skip_until == checkpoint_name:
                    skip_until = None
                if stop_at == checkpoint_name:
                    break
                continue

            if skip_until:
                continue

            if isinstance(stage, list):
                self._run_extractor_stage(stage, scenes, result)
            elif self._is_processor_stage(stage):
                self._run_processor_stage(stage, result)
            else:
                raise TypeError(
                    f"Unsupported pipeline stage type: {type(stage).__name__}"
                )

        if effective_resume and skip_until:
            raise ValueError(
                f"Checkpoint '{effective_resume}' not found in pipeline stages"
            )

        if self.config.validate:
            self._validate(result)

        return result

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

    def _initialize_result(
        self,
        scenes: list[dict],
        metadata: Mapping[int, Mapping[str, Any]],
        context: Mapping[str, Any] | None,
        load_checkpoint: Path | None,
    ) -> tuple[PipelineResult, str | None]:
        normalized_metadata = {int(k): dict(v) for k, v in (metadata or {}).items()}

        if load_checkpoint:
            result = PipelineResult.load(load_checkpoint)
            if normalized_metadata:
                result.metadata.update(normalized_metadata)
            if context:
                result.context.update(dict(context))
            return result, load_checkpoint.stem

        if not scenes:
            raise ValueError("Pipeline requires at least one scene")

        result = PipelineResult()
        result.scenes = [dict(item) for item in scenes]
        result.metadata = normalized_metadata
        if context:
            result.context.update(dict(context))
        return result, None

    def _register_extractor_stage(
        self, extractors: Sequence[ClueExtractor]
    ) -> None:
        for extractor in extractors:
            for member in extractor.registry_members():
                if member.clue_type not in self._registry:
                    self._registry.register(member)

    def _run_extractor_stage(
        self,
        extractors: list[ClueExtractor],
        scenes: list[dict],
        result: PipelineResult,
    ) -> None:
        if not scenes:
            raise ValueError("Extractor stage requires at least one scene")

        entries: list[tuple[int, str]] = [
            (int(entry["scene"]), str(entry["text"])) for entry in scenes
        ]

        for extractor in extractors:
            extractor.configure(self.config)

            if isinstance(extractor, BatchExtractor):
                if not entries:
                    continue
                batch_size = max(1, extractor.effective_batch_size())
                for idx in range(0, len(entries), batch_size):
                    batch = entries[idx : idx + batch_size]
                    clues = extractor.batch_extract(batch)
                    self._merge_extractor_output(extractor, clues, result)

                    participants = extractor.participants()
                    for scene_id, _ in batch:
                        names = participants.get(scene_id)
                        if names:
                            result.put_participants(scene_id, names)
            else:
                for scene_id, text in entries:
                    clues = extractor.extract(text, scene_id)
                    result.append_clues(extractor.clue_type, clues)

    def _run_processor_stage(
        self, processor: Processor, result: PipelineResult
    ) -> None:
        processor.configure(self.config)
        output = processor(result)
        if output is not None:
            result.put_output(output)

    @staticmethod
    def _is_processor_stage(stage: Any) -> TypeGuard[Processor]:
        return (
            hasattr(stage, "configure")
            and callable(getattr(stage, "configure"))
            and callable(getattr(stage, "__call__", None))
            and hasattr(stage, "checkpoint_id")
            and callable(getattr(stage, "checkpoint_id"))
            and hasattr(stage, "result_type")
        )

    def _validate(self, result: PipelineResult) -> None:
        validator = ValidationPipeline(self._registry)
        known_scenes = {int(item["scene"]) for item in result.scenes}
        context = ValidationContext(known_scenes=known_scenes)
        validations = validator.validate_batch(result.all_clues, context=context)
        result.validation = validations

        if self.config.strict_validation:
            errors: list[str] = []
            for clue, records in validations:
                for record in records:
                    if not record.passed:
                        errors.append(
                            f"Scene {clue.scene} | {clue.id} ({clue.clue_type}) "
                            f"failed {record.level}: {record.errors}"
                        )
            if errors:
                summary = f"Validation failed: {len(errors)} error(s) found"
                raise ValueError(summary + "\n" + "\n".join(errors))

    def _merge_extractor_output(
        self,
        extractor: ClueExtractor,
        clues: Sequence[BaseClue],
        result: PipelineResult,
    ) -> None:
        if not clues:
            return

        if isinstance(extractor, CombinedBatchExtractor):
            from collections import defaultdict

            buckets: dict[type[BaseClue], list[BaseClue]] = defaultdict(list)
            for clue in clues:
                buckets[type(clue)].append(clue)
            for clue_type, items in buckets.items():
                result.append_clues(clue_type, items)
        else:
            result.append_clues(extractor.clue_type, clues)
