from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, Type

from pydantic import BaseModel, Field

from framework.core.base import ClueValidator
from framework.core.batch import BatchExtractor
from framework.schema import EvidenceClippingMixin, PairClue, ValidationResult
from framework.utils import parse_model

if TYPE_CHECKING:
    from framework.core.pipeline import PipelineConfig


class ToMValidator(ClueValidator):
    def validate_semantic(self, clue: "ToMClue") -> ValidationResult:
        if not clue.claim:
            return ValidationResult.fail(
                level="semantic", errors=["claim must be non-empty"]
            )
        return ValidationResult.ok(level="semantic")

    def validate_coherence(
        self, clue: "ToMClue", context: Mapping[str, object] | None = None
    ) -> ValidationResult | None:
        _ = clue, context
        return None


class ToMExtractor(BatchExtractor):
    _clue_slug = "tom"

    @property
    def clue_type(self) -> type["ToMClue"]:  # noqa: D401
        return ToMClue

    def configure(self, config: "PipelineConfig") -> None:
        super().configure(config)
        if self._client is None:
            self._client = config.client
        if self._batch_size is None:
            self._batch_size = config.batch_size or self.batch_size
        if self._client is None:
            raise ValueError("ToMExtractor requires a client; none provided in config")

    def get_clue_specification(self) -> dict:
        return {
            "clue_type": "tom",
            "display_name": "THEORY-OF-MIND CLUES",
            "purpose": "Mental states that one character holds ABOUT another character.",
            "concepts": [
                ("BelievesAbout", "What character A believes regarding character B."),
                (
                    "FeelsTowards",
                    "Emotional stance character A has towards character B.",
                ),
                ("IntendsTo", "What character A plans to do regarding character B."),
                ("DesiresFor", "What character A wants from or for character B."),
            ],
            "special_rules": [
                "Signals must be explicit in dialogue, narration, or SDH cues—never inferred.",
                "Pairs must be exact character names (thinker, target).",
                "Omit entries if the target is ambiguous or only implied.",
            ],
            "schema_model": ToMClueAPI,
        }

    def _parse_response(
        self, raw_payload: Any, segment_id: int
    ) -> tuple[list[str], list["ToMClue"]]:
        schema_model = self._build_response_schema()
        payload_model = parse_model(schema_model, raw_payload)
        participants = list(getattr(payload_model, "participants", []))
        tom_items = getattr(payload_model, "tom_clues", [])

        clues: list[ToMClue] = []
        for item in tom_items:
            clue_api = (
                item
                if isinstance(item, ToMClueAPI)
                else ToMClueAPI.model_validate(item)
            )
            clues.append(clue_api.to_internal())
        return participants, clues

    def get_api_model(self) -> Type[BaseModel]:
        return ToMClueAPI

    def validator(self) -> ClueValidator:
        return ToMValidator()

    def score(self, clue: "ToMClue") -> float:
        _ = clue
        return 0.0


ToMKind = Literal["BelievesAbout", "FeelsTowards", "IntendsTo", "DesiresFor"]


class ToMClue(PairClue):
    clue_type: Literal["tom"] = "tom"
    kind: ToMKind
    claim: str


class ToMClueAPI(EvidenceClippingMixin):
    id: str | None = None
    segment: int
    pair: list[str] = Field(min_length=2, max_length=2)
    clue_type: Literal["tom"] = "tom"
    evidence: str
    kind: ToMKind
    claim: str

    def to_internal(self) -> ToMClue:
        data = self.model_dump()
        data["pair"] = tuple(data["pair"])
        data["id"] = data.get("id") or ""
        return ToMClue.model_validate(data)


__all__ = ["ToMExtractor", "ToMValidator", "ToMClue", "ToMClueAPI", "ToMKind"]
