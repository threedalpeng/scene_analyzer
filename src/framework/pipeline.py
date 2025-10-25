from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, Type

from framework.base import ClueExtractor
from framework.registry import ClueRegistry
from framework.result import PipelineResult
from framework.validation import ValidationContext, ValidationPipeline
from schema import BaseClue, ValidationResult


Processor = Callable[[PipelineResult], Any]


@dataclass(slots=True)
class PipelineConfig:
    client: Any | None = None
    batch_size: int = 10
    validate: bool = True
    strict_validation: bool = False
    checkpoint_enabled: bool = False
    output_dir: Path | None = None

    def as_mapping(self) -> Mapping[str, Any]:
        return {
            "client": self.client,
            "batch_size": self.batch_size,
            "validate": self.validate,
            "strict_validation": self.strict_validation,
            "checkpoint_enabled": self.checkpoint_enabled,
            "output_dir": self.output_dir,
        }


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        client: Any | None = None,
        batch_size: int = 10,
        validate: bool = True,
        strict_validation: bool = False,
        checkpoint_enabled: bool = False,
        output_dir: Path | None = None,
    ) -> None:
        if isinstance(config, PipelineConfig):
            base_config = config
        else:
            base_config = PipelineConfig(
                client=client,
                batch_size=batch_size,
                validate=validate,
                strict_validation=strict_validation,
                checkpoint_enabled=checkpoint_enabled,
                output_dir=output_dir,
            )
        self.config = base_config
        self._extractors: list[ClueExtractor] = []
        self._processors: list[Processor] = []

    def extract(
        self,
        extractor: Type[ClueExtractor] | ClueExtractor,
        /,
        **params: Any,
    ) -> "Pipeline":
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
            registry.register(extractor)

        scene_pairs: list[tuple[int, str]] = [
            (int(item["scene"]), str(item["text"])) for item in scenes
        ]

        for extractor in self._extractors:
            clues = list(extractor.batch_extract(scene_pairs))
            result.put_clues(extractor.clue_type, clues)
            participants = getattr(extractor, "participants", None)
            if callable(participants):
                for scene_id, names in dict(participants()).items():
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
            configure = getattr(extractor, "configure", None)
            if callable(configure):
                configure(self.config)

    def _configure_processors(self) -> None:
        for processor in self._processors:
            configure = getattr(processor, "configure", None)
            if callable(configure):
                configure(self.config)

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
        validations: Sequence[tuple[BaseClue, Sequence[ValidationResult]]]
    ) -> None:
        errors: list[str] = []
        for clue, records in validations:
            for record in records:
                if not getattr(record, "passed", True):
                    errors.append(
                        f"{clue.id} ({clue.clue_type}) failed {record.level}: {record.errors}"
                    )
        if errors:
            raise ValueError("Validation failed:\n" + "\n".join(errors))
