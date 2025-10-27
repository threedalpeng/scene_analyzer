from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Type

from google import genai

from framework.base import ClueExtractor
from framework.batch import BatchExtractor, CombinedBatchExtractor
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
        _ = resume
        self._configure_extractors()
        self._configure_processors()

        result = PipelineResult()
        result.metadata = {int(k): dict(v) for k, v in metadata.items()}
        result.scenes = [dict(item) for item in scenes]
        if context:
            result.context.update(dict(context))

        registry = ClueRegistry()
        for extractor in self._extractors:
            for member in extractor.registry_members():
                registry.register(member)

        scene_pairs: list[tuple[int, str]] = [
            (int(item["scene"]), str(item["text"])) for item in scenes
        ]

        for extractor in self._extractors:
            if isinstance(extractor, CombinedBatchExtractor):
                mixed_clues = list(extractor.batch_extract(scene_pairs))
                buckets: dict[type[BaseClue], list[BaseClue]] = {}
                for clue in mixed_clues:
                    buckets.setdefault(type(clue), []).append(clue)
                for clue_type, clues in buckets.items():
                    result.append_clues(clue_type, clues)
            else:
                clues = list(extractor.batch_extract(scene_pairs))
                result.put_clues(extractor.clue_type, clues)
            for scene_id, names in extractor.participants().items():
                result.put_participants(scene_id, names)

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

        for processor in self._processors:
            output = processor(result)
            if output is not None:
                result.put_output(output)

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
