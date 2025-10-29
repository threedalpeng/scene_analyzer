from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, Type, TypeGuard, cast

from google import genai

from framework.core.base import ClueExtractor
from framework.core.batch import BatchExtractor, CombinedBatchExtractor
from framework.core.processor import Processor
from framework.core.registry import ClueRegistry
from framework.core.result import PipelineResult
from framework.core.validation import ValidationContext, ValidationPipeline
from framework.schema import BaseClue


@dataclass(slots=True)
class PipelineConfig:
    client: genai.Client | None = None
    batch_size: int = 50
    strict_validation: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any | None = None) -> Any | None:
        return self.extras.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.extras[key] = value


CheckpointMarker = tuple[Literal["checkpoint"], str]
ValidationMarker = tuple[Literal["validate"], bool | None]


class Pipeline:
    """
    Coordinate clue extraction, post-processing, and validation stages.

    A pipeline is configured by chaining extractor, processor, and optional
    checkpoint or validation markers before invoking :meth:`run`. Stages execute
    sequentially over the provided segment list, sharing state through a
    :class:`PipelineResult` instance.

    Examples:
        >>> pipeline = Pipeline(PipelineConfig())
        >>> pipeline.extract([ActExtractor, ToMExtractor])
        ...         .process(AliasResolver())
        ...         .validate(strict=True)
        >>> result = pipeline.run(
        ...     segments=[{"segment": 1, "text": "Scene 1"}],
        ... )
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._stages: list[
            list[ClueExtractor] | Processor | CheckpointMarker | ValidationMarker
        ] = []
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
        """
        Register one or more clue extractor stages.

        Args:
            extractor: Extractor class, instance, or list of batch extractors.
            **params: Keyword arguments forwarded to the extractor constructor
                when a class is supplied.

        Returns:
            The pipeline instance (for fluent chaining).

        Raises:
            ValueError: If keyword parameters are provided for a list input.
            TypeError: If a combined list contains non-batch extractors.
        """
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
        """
        Append a processor stage that operates on accumulated results.

        Args:
            processor: Processor instance to append.

        Returns:
            The pipeline instance to enable chaining.
        """
        self._stages.append(processor)
        return self

    def pipe(self, other: "Pipeline") -> "Pipeline":
        """
        Append all stages from another pipeline instance.

        The supplied pipeline is not mutated; shallow copies of extractor stage
        lists are taken to avoid shared state between pipelines.
        """
        for stage in other._stages:
            if isinstance(stage, list):
                new_stage = list(stage)
                self._register_extractor_stage(new_stage)
                self._stages.append(new_stage)
            elif self._is_processor_stage(stage):
                self._stages.append(stage)
            elif isinstance(stage, tuple) and stage[0] in {"checkpoint", "validate"}:
                self._stages.append(stage)
            else:
                raise TypeError(
                    f"Unsupported pipeline stage type: {type(stage).__name__}"
                )

        return self

    def checkpoint(self, name: str) -> "Pipeline":
        """Insert a named checkpoint marker into the execution plan."""
        if not name:
            raise ValueError("Checkpoint name must be a non-empty string")
        self._stages.append(("checkpoint", name))
        return self

    def validate(self, *, strict: bool | None = None) -> "Pipeline":
        """
        Insert a validation marker executed after preceding stages.

        Args:
            strict: Override for strict-mode enforcement. If ``None``, the
                pipeline configuration decides whether validation errors raise.
        """
        self._stages.append(("validate", strict))
        return self

    def run(
        self,
        segments: list[dict],
        *,
        context: Mapping[str, Any] | None = None,
        checkpoint_dir: Path | None = None,
        load_checkpoint: Path | None = None,
        resume_from: str | None = None,
        stop_at: str | None = None,
        auto_save: bool = False,
    ) -> PipelineResult:
        """
        Execute the configured pipeline over the provided segments.

        Args:
            segments: List of segment dictionaries. Each entry must include
                ``segment`` (int identifier) and ``text`` (string content). Any
                additional metadata can be stored alongside these keys.
            context: Optional global context dictionary shared across stages.
            checkpoint_dir: Directory used for checkpoint persistence when
                ``auto_save`` is enabled.
            load_checkpoint: Resume from a serialized :class:`PipelineResult`.
            resume_from: Skip execution until the named checkpoint is reached.
            stop_at: Halt execution after the named checkpoint completes.
            auto_save: When ``True``, automatically persist checkpoints to
                ``checkpoint_dir``.

        Returns:
            :class:`PipelineResult` containing accumulated clues, processor
            outputs, validation reports, failures, and context.

        Raises:
            ValueError: If no segments are provided and no checkpoint is loaded.
            ValueError: If ``resume_from`` references an unknown checkpoint.
            RuntimeError: If strict validation is enabled and errors are found.
        """
        if not segments and load_checkpoint is None:
            raise ValueError("Pipeline requires at least one segment")

        result, derived_resume = self._initialize_result(
            segments, context, load_checkpoint
        )
        effective_resume = resume_from or derived_resume

        skip_until = effective_resume
        for stage in self._stages:
            if isinstance(stage, tuple):
                kind = stage[0]
                if kind == "checkpoint":
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

                if kind == "validate":
                    strict_flag = cast(bool | None, stage[1])
                    self._run_validation_stage(strict_flag, result)
                    continue

                raise TypeError(f"Unsupported pipeline marker: {kind}")

            if skip_until:
                continue

            if isinstance(stage, list):
                self._run_extractor_stage(stage, segments, result)
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
        segments: list[dict],
        context: Mapping[str, Any] | None,
        load_checkpoint: Path | None,
    ) -> tuple[PipelineResult, str | None]:
        if load_checkpoint:
            result = PipelineResult.load(load_checkpoint)
            if context:
                result.merge_context(dict(context))
            return result, load_checkpoint.stem

        if not segments:
            raise ValueError("Pipeline requires at least one segment")

        result = PipelineResult(segments=[dict(item) for item in segments])
        if context:
            result.merge_context(dict(context))
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
        segments: list[dict],
        result: PipelineResult,
    ) -> None:
        if not segments:
            raise ValueError("Extractor stage requires at least one segment")

        entries: list[tuple[int, str]] = [
            (int(entry["segment"]), str(entry["text"])) for entry in segments
        ]

        for extractor in extractors:
            component_name = extractor.__class__.__name__
            try:
                extractor.configure(self.config)
            except Exception as err:
                result.record_failure(0, "extraction", component_name, err)
                raise
            extractor.set_failure_recorder(
                lambda segment_id, message, component=component_name: result.record_failure(
                    segment_id, "extraction", component, message
                )
            )
            try:
                if isinstance(extractor, BatchExtractor):
                    if not entries:
                        continue
                    batch_size = max(1, extractor.effective_batch_size())
                    for idx in range(0, len(entries), batch_size):
                        batch = entries[idx : idx + batch_size]
                        try:
                            clues = extractor.batch_extract(batch)
                        except Exception as err:
                            result.record_failure(0, "extraction", component_name, err)
                            raise
                        self._merge_extractor_output(extractor, clues, result)

                    self._store_participants(result, extractor.participants())
                else:
                    for segment_id, text in entries:
                        try:
                            clues = extractor.extract(text, segment_id)
                        except Exception as err:
                            result.record_failure(
                                segment_id, "extraction", component_name, err
                            )
                            raise
                        result.add_clues(extractor.clue_type, clues)
                    self._store_participants(result, extractor.participants())
            finally:
                extractor.set_failure_recorder(None)

    def _run_processor_stage(
        self, processor: Processor, result: PipelineResult
    ) -> None:
        try:
            processor.configure(self.config)
        except Exception as err:
            result.record_failure(0, "processing", processor.__class__.__name__, err)
            raise
        try:
            output = processor(result)
        except Exception as err:
            result.record_failure(0, "processing", processor.__class__.__name__, err)
            raise
        if output is not None:
            result.put_output(output)

    @staticmethod
    def _is_processor_stage(stage: Any) -> TypeGuard[Processor]:
        return isinstance(stage, Processor)

    def _run_validation_stage(
        self, strict: bool | None, result: PipelineResult
    ) -> None:
        validator = ValidationPipeline(self._registry)
        known_segments = {int(item["segment"]) for item in result.segments}
        known_clue_ids = {clue.id for clue in result.all_clues if getattr(clue, "id", None)}
        context = ValidationContext(
            known_segments=known_segments, known_clue_ids=known_clue_ids
        )
        validations = validator.validate_batch(result.all_clues, context=context)
        result.validation = validations

        enforce_strict = self.config.strict_validation if strict is None else strict
        if enforce_strict:
            errors: list[str] = []
            for clue, records in validations:
                for record in records:
                    if not record.passed:
                        errors.append(
                            f"Segment {clue.segment} | {clue.id} ({clue.clue_type}) "
                            f"failed {record.level}: {record.errors}"
                        )
            if errors:
                summary = f"Validation failed: {len(errors)} error(s) found"
                raise ValueError(summary + "\n" + "\n".join(errors))

    @staticmethod
    def _store_participants(
        result: PipelineResult, participants: Mapping[int, Sequence[str]]
    ) -> None:
        if not participants:
            return

        key = "framework.participants"
        existing = result.context.get(key)
        if not isinstance(existing, dict):
            existing = {}
            result.context[key] = existing

        for segment_id, names in participants.items():
            if not names:
                continue
            bucket = existing.setdefault(int(segment_id), [])
            for name in names:
                normalized = name.strip()
                if not normalized or normalized in bucket:
                    continue
                bucket.append(normalized)

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
                result.add_clues(clue_type, items)
        else:
            result.add_clues(extractor.clue_type, clues)
