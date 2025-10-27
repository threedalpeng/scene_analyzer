from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, Type

from pydantic import BaseModel, Field

from framework.base import ClueValidator
from framework.batch import BatchExtractor
from schema import BaseClue, EvidenceClippingMixin, ValidationResult
from utils import parse_model

if TYPE_CHECKING:
    from framework.pipeline import PipelineConfig


class TemporalValidator(ClueValidator):
    def validate_semantic(self, clue: "TemporalClue") -> ValidationResult:
        _ = clue
        return ValidationResult.ok(level="semantic")

    def validate_coherence(
        self, clue: "TemporalClue", context: Mapping[str, object] | None = None
    ) -> ValidationResult | None:
        if context is None:
            return None
        references = getattr(clue, "references_scenes", [])
        known = context.get("known_scenes", set())
        known_set = set(known) if isinstance(known, (set, list, tuple)) else set()
        missing = [ref for ref in references if ref not in known_set]
        if missing:
            return ValidationResult.ok(
                level="coherence",
                warnings=[
                    "temporal clue references unknown scene ids: "
                    + ", ".join(str(m) for m in missing)
                ],
            )
        return None


class TemporalExtractor(BatchExtractor):
    _clue_slug = "temporal"

    @property
    def clue_type(self) -> type["TemporalClue"]:  # noqa: D401
        return TemporalClue

    def configure(self, config: "PipelineConfig") -> None:
        super().configure(config)
        if self._client is None:
            self._client = config.client
        if self._batch_size is None:
            self._batch_size = config.batch_size or self.batch_size
        if self._client is None:
            raise ValueError(
                "TemporalExtractor requires a client; none provided in config"
            )

    def get_clue_specification(self) -> dict:
        return {
            "clue_type": "temporal",
            "display_name": "TEMPORAL CLUES",
            "purpose": "Chronological relationships between scenes to reconstruct fabula from syuzhet.",
            "concepts": [
                (
                    "References_scenes",
                    "Which earlier scenes this scene explicitly references or continues.",
                ),
                (
                    "Time_offset",
                    "Approximate time delta from the narrative present (days, negative for past).",
                ),
                ("Is_flashback", "True when the scene is presented out of chronological order."),
            ],
            "special_rules": [
                "Use only explicit temporal markers (timestamps, dialogue references, SDH cues).",
                "If uncertain about the chronological relationship, omit the clue.",
            ],
            "schema_model": TemporalClueAPI,
        }

    def _parse_response(
        self, raw_payload: Any, scene_id: int
    ) -> tuple[list[str], list["TemporalClue"]]:
        schema_model = self._build_response_schema()
        payload_model = parse_model(schema_model, raw_payload)
        participants = list(getattr(payload_model, "participants", []))
        temporal_items = getattr(payload_model, "temporal_clues", [])

        clues: list[TemporalClue] = []
        for item in temporal_items:
            clue_api = (
                item
                if isinstance(item, TemporalClueAPI)
                else TemporalClueAPI.model_validate(item)
            )
            clues.append(clue_api.to_internal())
        return participants, clues

    def get_api_model(self) -> Type[BaseModel]:
        return TemporalClueAPI

    def score(self, clue: "TemporalClue") -> float:
        _ = clue
        return 0.0

    def validator(self) -> ClueValidator:
        return TemporalValidator()


class TemporalClue(BaseClue):
    clue_type: Literal["temporal"] = "temporal"
    references_scenes: list[int] = Field(default_factory=list)
    time_offset: int | None = None
    is_flashback: bool


class TemporalClueAPI(EvidenceClippingMixin):
    id: str | None = None
    scene: int
    clue_type: Literal["temporal"] = "temporal"
    evidence: str
    references_scenes: list[int] = Field(default_factory=list)
    time_offset: int | None = None
    is_flashback: bool

    def to_internal(self) -> TemporalClue:
        data = self.model_dump()
        data["id"] = data.get("id") or ""
        return TemporalClue.model_validate(data)


__all__ = ["TemporalExtractor", "TemporalClue", "TemporalClueAPI"]
