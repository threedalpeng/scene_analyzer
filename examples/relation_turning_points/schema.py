from typing import Literal

from pydantic import BaseModel, Field, field_validator

from framework.schema import BaseClue

Valence = Literal["positive", "negative", "neutral", "mixed"]
Durability = Literal["momentary", "temporary", "persistent"]
Salience = Literal["low", "medium", "high"]
Stakes = Literal["minor", "moderate", "major"]
Volition = Literal["voluntary", "coerced", "accidental"]
GoalAlignment = Literal["aligned", "orthogonal", "opposed"]
TurningCategory = Literal["Reversal", "Strengthening", "Commitment", "Dissolution"]


class EvidenceClippingMixin(BaseModel):
    """Mixin that enforces evidence clipping and trimming."""

    @field_validator("evidence", check_fields=False)
    @classmethod
    def _clip_api_evidence(cls, value: str) -> str:
        value = value.strip()
        return value if len(value) <= 200 else value[:200]


class PairClue(BaseClue):
    pair: tuple[str, str]

    @field_validator("pair")
    @classmethod
    def _sort_pair(cls, value: tuple[str, str]) -> tuple[str, str]:
        if len(value) != 2:
            raise ValueError("pair must contain exactly two entries")
        a, b = sorted(value)
        return a, b


class RelationState(BaseModel):
    valence: Valence
    durability: Durability
    label: str | None = None


class FinalRelation(BaseModel):
    valence: Valence
    label: str | None = None


class TurningEntry(BaseModel):
    segment: int
    category: TurningCategory
    pattern: str
    summary: str
    quote: str
    representative_act_id: str
    caused_by_tom_ids: list[str] = Field(default_factory=list)
    pre_state: RelationState
    post_state: RelationState


class DyadFinal(BaseModel):
    pair: tuple[str, str]
    turning_timeline: list[TurningEntry]
    final_relation: FinalRelation
    state_timeline_ref: str | None = None


class LLMEventRole(BaseModel):
    segment: int
    id: str
    event_role: Literal["turning", "supporting"]


class LLMAdjudication(BaseModel):
    event_roles: list[LLMEventRole]
    turning_timeline: list[TurningEntry]
    final_relation: FinalRelation
    log: list[str] = Field(default_factory=list)


class AliasGroup(BaseModel):
    canonical: str
    aliases: list[str]


class AliasGroups(BaseModel):
    groups: list[AliasGroup]
