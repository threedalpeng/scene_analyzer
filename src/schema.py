from typing import Literal

from pydantic import BaseModel, Field, field_validator

ToMKind = Literal["BelievesAbout", "FeelsTowards", "IntendsTo", "DesiresFor"]
StanceCore = Literal["cooperation", "hostility"]
Stance = Literal["cooperation", "hostility", "mixed", "neutral"]
Durability = Literal["momentary", "temporary", "persistent"]


class Axes(BaseModel):
    salience: Literal["low", "medium", "high"]
    stakes: Literal["minor", "moderate", "major"]
    durability: Durability
    volition: Literal["voluntary", "coerced", "accidental"]
    goal_alignment: Literal["aligned", "orthogonal", "opposed"]
    consequence_refs: list[int] = Field(default_factory=list)


class BaseClue(BaseModel):
    id: str
    scene: int
    pair: tuple[str, str]
    modality: Literal["tom", "act"]
    evidence: str

    @field_validator("pair")
    @classmethod
    def _sort_pair(cls, v: tuple[str, str]) -> tuple[str, str]:
        if len(v) != 2:
            raise ValueError("pair must contain exactly two entries")
        a, b = sorted(v)
        return a, b

    @field_validator("evidence")
    @classmethod
    def _clip_evidence(cls, v: str) -> str:
        v = v.strip()
        return v if len(v) <= 200 else v[:200]


class ToMClue(BaseClue):
    modality: Literal["tom"] = "tom"
    kind: ToMKind
    claim: str


class ActClue(BaseClue):
    modality: Literal["act"] = "act"
    actors: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    stance: StanceCore
    subtype: str
    axes: Axes


class IntraScenePayload(BaseModel):
    participants: list[str]
    tom_clues: list[ToMClue] = Field(default_factory=list)
    act_clues: list[ActClue] = Field(default_factory=list)


class BaseClueAPI(BaseModel):
    id: str
    scene: int
    pair: list[str] = Field(min_length=2, max_length=2)
    modality: Literal["tom", "act"]
    evidence: str


class ToMClueAPI(BaseClueAPI):
    modality: Literal["tom"] = "tom"
    kind: ToMKind
    claim: str

    def to_internal(self) -> ToMClue:
        data = self.model_dump()
        data["pair"] = tuple(data["pair"])
        return ToMClue.model_validate(data)


class ActClueAPI(BaseClueAPI):
    modality: Literal["act"] = "act"
    actors: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    stance: StanceCore
    subtype: str
    axes: Axes

    def to_internal(self) -> ActClue:
        data = self.model_dump()
        data["pair"] = tuple(data["pair"])
        return ActClue.model_validate(data)


class IntraScenePayloadAPI(BaseModel):
    participants: list[str]
    tom_clues: list[ToMClueAPI] = Field(default_factory=list)
    act_clues: list[ActClueAPI] = Field(default_factory=list)

    def to_internal(self) -> IntraScenePayload:
        return IntraScenePayload(
            participants=self.participants,
            tom_clues=[t.to_internal() for t in self.tom_clues],
            act_clues=[a.to_internal() for a in self.act_clues],
        )


# class WithEventRole(BaseModel):
#     event_role: Literal["turning", "supporting"]


# class ToMClueWithRole(ToMClue, WithEventRole): ...


# class ActClueWithRole(ActClue, WithEventRole): ...


class RelationState(BaseModel):
    stance: Stance
    durability: Durability


class TurningEntry(BaseModel):
    scene: int
    type: str  # betrayal|reconciliation|coerced_hostility|aligned_mission|revealed_identity|implicit_turn
    summary: str
    quote: str
    representative_act_id: str
    caused_by_tom_ids: list[str] = Field(default_factory=list)
    pre_state: RelationState
    post_state: RelationState


class DyadFinal(BaseModel):
    pair: tuple[str, str]
    final_relation: Stance
    turning_timeline: list[TurningEntry]
    state_timeline_ref: str | None = None


class LLMEventRole(BaseModel):
    scene: int
    id: str
    event_role: Literal["turning", "supporting"]


class LLMAdjudication(BaseModel):
    event_roles: list[LLMEventRole]
    turning_timeline: list[TurningEntry]
    final_relation: Stance
    log: list[str] = Field(default_factory=list)


class AliasGroup(BaseModel):
    canonical: str
    aliases: list[str]


class AliasGroups(BaseModel):
    groups: list[AliasGroup]
