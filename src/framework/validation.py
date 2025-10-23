from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from clues.act import ActClue
from clues.base import ClueExtractor
from clues.entity import EntityClue
from clues.registry import ClueRegistry
from clues.temporal import TemporalClue
from schema import BaseSignal, ValidationResult


@dataclass(slots=True)
class ValidationContext:
    """Context for coherence validation."""

    known_scenes: set[int]
    representative_act_ids: set[str]


class ValidationPipeline:
    """Apply structural → semantic → coherence validation layers."""

    def __init__(self, registry: ClueRegistry | None = None) -> None:
        self._registry = registry

    def _get_validator(self, signal: BaseSignal) -> ClueExtractor | None:
        if self._registry is None:
            return None
        try:
            return self._registry.get(signal.modality)
        except KeyError:
            return None

    def validate_signal(
        self, signal: BaseSignal, *, context: ValidationContext | None = None
    ) -> Sequence[ValidationResult]:
        results: list[ValidationResult] = []

        structural = self._validate_structural(signal)
        results.append(structural)
        if not structural.passed:
            return results

        validator = self._get_validator(signal)
        if validator is not None:
            semantic = validator.validator().validate_semantic(signal)
            results.append(semantic)
            if not semantic.passed:
                return results

        if context is not None:
            results.append(self._validate_coherence(signal, context))

        return results

    def validate_batch(
        self, signals: Iterable[BaseSignal], *, context: ValidationContext | None = None
    ) -> list[tuple[BaseSignal, Sequence[ValidationResult]]]:
        return [
            (signal, self.validate_signal(signal, context=context))
            for signal in signals
        ]

    @staticmethod
    def _validate_structural(signal: BaseSignal) -> ValidationResult:
        errors: list[str] = []
        if not signal.id:
            errors.append("id is required")
        if not signal.modality:
            errors.append("modality is required")
        if signal.scene is None or int(signal.scene) < 0:
            errors.append("scene must be non-negative")
        if not signal.evidence:
            errors.append("evidence is required")
        if not signal.id.startswith(f"{signal.modality}_"):
            errors.append("id must start with modality prefix")

        if errors:
            return ValidationResult.fail(level="structural", errors=errors)
        return ValidationResult.ok(level="structural")

    @staticmethod
    def _validate_coherence(
        signal: BaseSignal, context: ValidationContext
    ) -> ValidationResult:
        warnings: list[str] = []

        if isinstance(signal, ActClue):
            missing = [
                ref for ref in signal.axes.consequence_refs if ref not in context.known_scenes
            ]
            if missing:
                warnings.append(
                    "act consequence_refs reference unknown scenes: "
                    + ", ".join(str(m) for m in missing)
                )

        if isinstance(signal, TemporalClue):
            missing = [
                ref for ref in signal.references_scenes if ref not in context.known_scenes
            ]
            if missing:
                warnings.append(
                    "temporal reference to unknown scene ids: "
                    + ", ".join(str(m) for m in missing)
                )

        if isinstance(signal, EntityClue):
            if not signal.aliases_in_scene:
                warnings.append("entity clue should list aliases if present")

        return ValidationResult.ok(level="coherence", warnings=warnings)
