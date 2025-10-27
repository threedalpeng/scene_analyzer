from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Type

from google import genai

from framework.base import ClueExtractor
from framework.batch import BatchExtractor, CombinedBatchExtractor
from framework.checkpoint_session import CheckpointSession
from framework.processor import Processor
from framework.registry import ClueRegistry, ProcessorResultRegistry
from framework.result import PipelineResult
from framework.validation import ValidationContext, ValidationPipeline
from schema import BaseClue


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
        self._registry = ClueRegistry()
        self._result_registry = ProcessorResultRegistry()

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

        self._extractors.append(instance)
        for member in instance.registry_members():
            self._registry.register(member)
        return self

    def process(self, processor: Processor) -> "Pipeline":
        self._processors.append(processor)
        result_type = processor.result_type
        if result_type is not None and not self._result_registry.has_type(
            result_type.__name__
        ):
            self._result_registry.register(result_type)
        return self

    def run(
        self,
        scenes: list[dict],
        *,
        metadata: dict[int, dict] | None = None,
        context: Mapping[str, Any] | None = None,
        resume: bool = False,
    ) -> PipelineResult:
        if not scenes:
            raise ValueError("Pipeline requires at least one scene")

        metadata = metadata or {}
        session = self._create_session(resume)
        if session and not session.validate_inputs(scenes, metadata):
            raise ValueError("Input mismatch with checkpoint")

        result = self._initialize_result(session, scenes, metadata, context)

        self._run_extractors(scenes, result, session)
        self._run_processors(result, session)

        if self.config.validate:
            self._validate(result)

        if session:
            session.save_result(result)

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

    def _create_session(self, resume: bool) -> CheckpointSession | None:
        if not self.config.checkpoint_enabled or self.config.output_dir is None:
            return None
        checkpoint_dir = self.config.output_dir / "checkpoint"
        return CheckpointSession(checkpoint_dir, resume=resume)

    def _initialize_result(
        self,
        session: CheckpointSession | None,
        scenes: list[dict],
        metadata: Mapping[int, Mapping[str, Any]],
        context: Mapping[str, Any] | None,
    ) -> PipelineResult:
        if session and session.can_resume():
            return session.load_result(self._registry, self._result_registry)

        result = PipelineResult()
        result.scenes = [dict(item) for item in scenes]
        result.metadata = {int(k): dict(v) for k, v in metadata.items()}
        if context:
            result.context.update(dict(context))
        return result

    def _run_extractors(
        self,
        scenes: list[dict],
        result: PipelineResult,
        session: CheckpointSession | None,
    ) -> None:
        for extractor in self._extractors:
            extractor.configure(self.config)
            extractor_id = extractor.checkpoint_id()

            pending: list[tuple[int, str]] = []
            for entry in scenes:
                scene_id = int(entry["scene"])
                if session and session.is_scene_completed(extractor_id, scene_id):
                    continue
                pending.append((scene_id, str(entry["text"])))

            if not pending:
                continue

            if isinstance(extractor, BatchExtractor):
                batch_size = max(1, extractor.effective_batch_size())
                for idx in range(0, len(pending), batch_size):
                    batch = pending[idx : idx + batch_size]
                    clues = extractor.batch_extract(batch)
                    self._merge_extractor_output(extractor, clues, result)

                    participants = extractor.participants()
                    completed_ids: list[int] = []
                    for scene_id, _ in batch:
                        names = participants.get(scene_id)
                        if names:
                            result.put_participants(scene_id, names)
                        completed_ids.append(scene_id)

                    if session:
                        session.mark_scenes_completed(extractor_id, completed_ids)
                        session.save_result(result)
            else:
                for scene_id, text in pending:
                    clues = extractor.extract(text, scene_id)
                    result.append_clues(extractor.clue_type, clues)
                    if session:
                        session.mark_scenes_completed(extractor_id, [scene_id])
                        session.save_result(result)

    def _run_processors(
        self,
        result: PipelineResult,
        session: CheckpointSession | None,
    ) -> None:
        for processor in self._processors:
            processor.configure(self.config)
            processor_id = processor.checkpoint_id()

            if session and session.is_processor_completed(processor_id):
                continue

            output = processor(result)
            if output is not None:
                result.put_output(output)

            if session:
                session.mark_processor_completed(processor_id)
                session.save_result(result)

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
