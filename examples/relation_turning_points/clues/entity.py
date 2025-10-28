from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, Type

from pydantic import BaseModel, Field

from framework.core.base import ClueValidator
from framework.core.batch import BatchExtractor
from framework.schema import BaseClue, EvidenceClippingMixin, ValidationResult
from framework.utils import parse_model

if TYPE_CHECKING:
    from framework.core.pipeline import PipelineConfig


class EntityValidator(ClueValidator):
    def validate_semantic(self, clue: "EntityClue") -> ValidationResult:
        if not clue.aliases_in_segment:
            return ValidationResult.fail(
                level="semantic", errors=["alias list must not be empty"]
            )
        return ValidationResult.ok(level="semantic")

    def validate_coherence(
        self, clue: "EntityClue", context: Mapping[str, object] | None = None
    ) -> ValidationResult | None:
        if clue.aliases_in_segment:
            return None
        return ValidationResult.ok(
            level="coherence",
            warnings=["entity clue should list aliases if present"],
        )


class EntityExtractor(BatchExtractor):
    _clue_slug = "entity"

    @property
    def clue_type(self) -> type["EntityClue"]:  # noqa: D401
        return EntityClue

    def configure(self, config: "PipelineConfig") -> None:
        super().configure(config)
        if self._client is None:
            self._client = config.client
        if self._batch_size is None:
            self._batch_size = config.batch_size or self.batch_size
        if self._client is None:
            raise ValueError(
                "EntityExtractor requires a client; none provided in config"
            )

    def get_clue_specification(self) -> dict:
        return {
            "clue_type": "entity",
            "display_name": "ENTITY ALIAS CLUES",
            "purpose": "Resolve explicit alias relationships for characters mentioned in the segment.",
            "concepts": [
                ("Name", "Primary name used for the character in this segment."),
                (
                    "Aliases_in_segment",
                    "Other names or titles explicitly referring to the same person within this segment.",
                ),
            ],
            "special_rules": [
                "Only capture aliases when both names appear in the segment and refer to the same person.",
                "Evidence must quote the text that links the alias to the canonical name.",
            ],
            "schema_model": EntityClueAPI,
        }

    def _parse_response(
        self, raw_payload: Any, segment_id: int
    ) -> tuple[list[str], list["EntityClue"]]:
        schema_model = self._build_response_schema()
        payload_model = parse_model(schema_model, raw_payload)
        participants = list(getattr(payload_model, "participants", []))
        entity_items = getattr(payload_model, "entity_clues", [])

        clues: list[EntityClue] = []
        for item in entity_items:
            clue_api = (
                item
                if isinstance(item, EntityClueAPI)
                else EntityClueAPI.model_validate(item)
            )
            clues.append(clue_api.to_internal())
        return participants, clues

    def get_api_model(self) -> Type[BaseModel]:
        return EntityClueAPI

    def score(self, clue: "EntityClue") -> float:
        _ = clue
        return 0.0

    def validator(self) -> ClueValidator:
        return EntityValidator()


class EntityClue(BaseClue):
    clue_type: Literal["entity"] = "entity"
    name: str
    aliases_in_segment: list[str] = Field(default_factory=list)


class EntityClueAPI(EvidenceClippingMixin):
    id: str | None = None
    segment: int
    clue_type: Literal["entity"] = "entity"
    evidence: str
    name: str
    aliases_in_segment: list[str] = Field(default_factory=list)

    def to_internal(self) -> EntityClue:
        data = self.model_dump()
        data["id"] = data.get("id") or ""
        return EntityClue.model_validate(data)


__all__ = [
    "EntityExtractor",
    "EntityValidator",
    "EntityClue",
    "EntityClueAPI",
]
