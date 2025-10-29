from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Mapping, Sequence, Type

from pydantic import BaseModel, Field, field_validator

from framework.core.base import ClueValidator
from framework.core.batch import BatchExtractor
from framework.schema import ValidationResult
from schema import (
    Durability,
    EvidenceClippingMixin,
    GoalAlignment,
    DirectedPairClue,
    Salience,
    Stakes,
    Volition,
)
from framework.utils import parse_model

if TYPE_CHECKING:
    from framework.core.pipeline import PipelineConfig

STAKE_RANK = {"major": 3, "moderate": 2, "minor": 1}
SALIENCE_RANK = {"high": 3, "medium": 2, "low": 1}
DURABILITY_RANK = {"persistent": 3, "temporary": 2, "momentary": 1}


def act_score(clue: "ActClue") -> int:
    return (
        100 * STAKE_RANK.get(clue.axes.stakes, 0)
        + 10 * SALIENCE_RANK.get(clue.axes.salience, 0)
        + DURABILITY_RANK.get(clue.axes.durability, 0)
    )


def explode_directed(acts: Sequence["ActClue"]) -> list["ActClue"]:
    """Acts are already directed; filter out self-referential entries."""
    return [act for act in acts if act.source.lower() != act.target.lower()]


def bundle_same_segment(acts: Sequence["ActClue"]) -> list["ActClue"]:
    buckets: dict[tuple[int, str, str, str, str], list[ActClue]] = defaultdict(list)
    for act in acts:
        buckets[(act.segment, act.source, act.target, act.valence, act.pattern)].append(
            act
        )

    representatives: list[ActClue] = []
    for items in buckets.values():
        items = list(items)
        items.sort(key=act_score, reverse=True)
        representatives.append(items[0])
    return representatives


class ActValidator(ClueValidator):
    def validate_semantic(self, clue: "ActClue") -> ValidationResult:
        warnings: list[str] = []
        if clue.source.strip().lower() == clue.target.strip().lower():
            return ValidationResult.fail(
                level="semantic",
                errors=["actor and target must be different individuals"],
            )
        if clue.axes.stakes == "major" and not clue.referenced_segments:
            warnings.append("major stakes usually reference downstream segments")
        return ValidationResult.ok(level="semantic", warnings=warnings)

    def validate_coherence(
        self, clue: "ActClue", context: Mapping[str, object] | None = None
    ) -> ValidationResult | None:
        _ = clue, context
        return None


class ActExtractor(BatchExtractor):
    _clue_slug = "act"

    @property
    def clue_type(self) -> type["ActClue"]:  # noqa: D401
        return ActClue

    def configure(self, config: "PipelineConfig") -> None:
        super().configure(config)
        if self._client is None:
            self._client = config.client
        if self._batch_size is None:
            self._batch_size = config.batch_size or self.batch_size
        if self._client is None:
            raise ValueError("ActExtractor requires a client; none provided in config")

    def get_clue_specification(self) -> dict:
        return {
            "clue_type": "act",
            "display_name": "ACT CLUES",
            "purpose": "Observable actions where one character (actor) affects another character (target).",
            "concepts": [
                (
                    "Action",
                    "Explicit behavior described in the segment where an actor affects a target.",
                ),
                (
                    "Valence",
                    "positive (helps, cooperates, supports) or negative (harms, opposes, threatens).",
                ),
                (
                    "Pattern",
                    "Descriptive label for the action type (e.g., 'rescues', 'betrays', 'shares_intel').",
                ),
                (
                    "Axes",
                    "Ratings for salience, stakes, durability, volition, and goal_alignment.",
                ),
                (
                    "Referenced_segments",
                    "Downstream segment ids where this action's consequences appear.",
                ),
            ],
            "special_rules": [
                "Direction is critical: every action must clearly identify actor → target.",
                "If multiple actors or targets exist, create separate action entries for each combination.",
                "Mark durability as 'persistent' for irreversible outcomes (pledge, affiliation change, death, etc.).",
                "List referenced_segments only when the downstream impact is explicitly stated or strongly implied.",
                "Omit the clue if you cannot confidently identify both actor and target from explicit text.",
            ],
            "schema_model": ActClueAPI,
        }

    def _parse_response(
        self, raw_payload: Any, segment_id: int
    ) -> tuple[list[str], list["ActClue"]]:
        schema_model = self._build_response_schema()
        payload_model = parse_model(schema_model, raw_payload)
        participants = list(getattr(payload_model, "participants", []))
        act_items = getattr(payload_model, "act_clues", [])

        clues: list[ActClue] = []
        for item in act_items:
            clue_api = (
                item
                if isinstance(item, ActClueAPI)
                else ActClueAPI.model_validate(item)
            )
            clues.extend(clue_api.to_internal())
        return participants, clues

    def get_api_model(self) -> Type[BaseModel]:
        return ActClueAPI

    def score(self, clue: "ActClue") -> float:
        return float(act_score(clue))

    def validator(self) -> ClueValidator:
        return ActValidator()


ValenceCore = Literal["positive", "negative"]


class Axes(BaseModel):
    salience: Salience
    stakes: Stakes
    durability: Durability
    volition: Volition
    goal_alignment: GoalAlignment


class ActClue(DirectedPairClue):
    source_field: ClassVar[str] = "source"
    target_field: ClassVar[str] = "target"
    clue_type: Literal["act"] = "act"
    source: str
    target: str
    valence: ValenceCore
    pattern: str
    axes: Axes


class ActClueAPI(EvidenceClippingMixin):
    id: str | None = None
    segment: int
    clue_type: Literal["act"] = "act"
    evidence: str
    actors: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    valence: ValenceCore
    pattern: str
    axes: Axes
    referenced_segments: list[int] = Field(default_factory=list)

    @field_validator("actors", "targets")
    @classmethod
    def _strip_list(cls, value: list[str]) -> list[str]:
        return [v.strip() for v in value if v and v.strip()]

    def to_internal(self) -> list[ActClue]:
        base = self.model_dump()
        base_id = (base.get("id") or "").strip()
        segment = base["segment"]
        pattern = base["pattern"]
        actors: list[str] = base.get("actors", []) or []
        targets: list[str] = base.get("targets", []) or []
        if not actors or not targets:
            raise ValueError("Act clue requires at least one actor and one target")

        combos: list[ActClue] = []
        for src in actors:
            for dst in targets:
                if src.lower() == dst.lower():
                    continue
                suffix = f"{src}->{dst}"
                if base_id:
                    clue_id = f"{base_id}__{suffix}"
                else:
                    clue_id = f"seg{segment}_{pattern}__{suffix}"
                data = {
                    "id": clue_id,
                    "segment": segment,
                    "clue_type": base["clue_type"],
                    "evidence": base["evidence"],
                    "references": base.get("references", []),
                    "referenced_segments": base.get("referenced_segments", []),
                    "source": src,
                    "target": dst,
                    "valence": base["valence"],
                    "pattern": base["pattern"],
                    "axes": base["axes"],
                }
                combos.append(ActClue.model_validate(data))
        return combos


__all__ = [
    "ActExtractor",
    "ActValidator",
    "ActClue",
    "ActClueAPI",
    "Axes",
    "ValenceCore",
    "act_score",
    "explode_directed",
    "bundle_same_segment",
]
