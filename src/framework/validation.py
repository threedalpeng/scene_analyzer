from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from framework.base import ClueExtractor
from framework.registry import ClueRegistry
from schema import BaseClue, ValidationResult


@dataclass(slots=True)
class ValidationContext:
    """Context for coherence validation."""

    known_scenes: set[int]


class ValidationPipeline:
    """Apply structural → semantic → coherence validation layers."""

    def __init__(self, registry: ClueRegistry | None = None) -> None:
        self._registry = registry

    def _get_validator(self, clue: BaseClue) -> ClueExtractor | None:
        if self._registry is None:
            return None
        clue_cls = type(clue)
        if not isinstance(clue_cls, type) or not issubclass(clue_cls, BaseClue):
            return None
        try:
            return self._registry.get(clue_cls)
        except KeyError:
            return None

    def validate_clue(
        self, clue: BaseClue, *, context: ValidationContext | None = None
    ) -> Sequence[ValidationResult]:
        results: list[ValidationResult] = []

        structural = self._validate_structural(clue)
        results.append(structural)
        if not structural.passed:
            return results

        validator = self._get_validator(clue)
        validator_obj = None
        if validator is not None:
            validator_obj = validator.validator()
            semantic = validator_obj.validate_semantic(clue)
            results.append(semantic)
            if not semantic.passed:
                return results

        if context is not None:
            context_payload: Mapping[str, object] = {
                "known_scenes": context.known_scenes,
            }
        else:
            context_payload = {}

        coherence = None
        if validator_obj is not None:
            coherence = validator_obj.validate_coherence(clue, context_payload)
            if coherence is not None:
                results.append(coherence)

        if coherence is None and context is not None:
            fallback = self._validate_default_coherence(clue, context_payload)
            if fallback is not None:
                results.append(fallback)

        return results

    def validate_batch(
        self, clues: Iterable[BaseClue], *, context: ValidationContext | None = None
    ) -> list[tuple[BaseClue, Sequence[ValidationResult]]]:
        return [(clue, self.validate_clue(clue, context=context)) for clue in clues]

    @staticmethod
    def _validate_structural(clue: BaseClue) -> ValidationResult:
        errors: list[str] = []
        if not clue.id:
            errors.append("id is required")
        if not clue.clue_type:
            errors.append("clue_type is required")
        if clue.scene is None or int(clue.scene) < 0:
            errors.append("scene must be non-negative")
        if not clue.evidence:
            errors.append("evidence is required")
        if not clue.id.startswith(f"{clue.clue_type}_"):
            errors.append("id must start with clue_type prefix")

        if errors:
            return ValidationResult.fail(level="structural", errors=errors)
        return ValidationResult.ok(level="structural")

    @staticmethod
    def _validate_default_coherence(
        clue: BaseClue, context: Mapping[str, object]
    ) -> ValidationResult | None:
        warnings: list[str] = []

        if clue.clue_type == "act":
            axes = getattr(clue, "axes", None)
            consequence_refs = getattr(axes, "consequence_refs", []) if axes else []
            known = context.get("known_scenes", set())
            missing = [ref for ref in consequence_refs if ref not in known]
            if missing:
                warnings.append(
                    "act clue consequence_refs reference unknown scenes: "
                    + ", ".join(str(m) for m in missing)
                )

        if clue.clue_type == "temporal":
            references = getattr(clue, "references_scenes", [])
            known = context.get("known_scenes", set())
            missing = [ref for ref in references if ref not in known]
            if missing:
                warnings.append(
                    "temporal clue references unknown scene ids: "
                    + ", ".join(str(m) for m in missing)
                )

        if clue.clue_type == "entity":
            aliases = getattr(clue, "aliases_in_scene", [])
            if not aliases:
                warnings.append("entity clue should list aliases if present")

        if warnings:
            return ValidationResult.ok(level="coherence", warnings=warnings)
        return None
