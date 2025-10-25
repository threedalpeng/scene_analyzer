from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, Type

from google.genai import types
from pydantic import BaseModel, Field, computed_field, field_validator

from framework.base import BatchExtractor, ClueValidator
from schema import (
    Durability,
    GoalAlignment,
    PairClue,
    Salience,
    Stakes,
    ValidationResult,
    Volition,
)
from utils import parse_model

if TYPE_CHECKING:
    from framework.pipeline import PipelineConfig

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
    out: list[ActClue] = []
    for act in acts:
        actors = act.actors or [act.pair[0]]
        targets = act.targets or [act.pair[1]]
        for src in actors:
            for dst in targets:
                if src.lower() == dst.lower():
                    continue
                out.append(
                    act.model_copy(
                        update={
                            "id": f"{act.id}__{src}->{dst}",
                            "actors": [src],
                            "targets": [dst],
                            "pair": tuple(sorted((src, dst))),
                        }
                    )
                )
    return out


def bundle_same_scene(acts: Sequence["ActClue"]) -> list["ActClue"]:
    buckets: dict[tuple[int, str, str, str, str], list[ActClue]] = defaultdict(list)
    for act in acts:
        src = act.actors[0] if act.actors else act.pair[0]
        dst = act.targets[0] if act.targets else act.pair[1]
        buckets[(act.scene, src, dst, act.valence, act.pattern)].append(act)

    representatives: list[ActClue] = []
    for items in buckets.values():
        items = list(items)
        items.sort(key=act_score, reverse=True)
        representatives.append(items[0])
    return representatives


ACT_PROMPT_RULES = """
CORE CONCEPTS:

Action: An observable behavior where one character (actor) does something that 
affects another character (target). Must be explicitly described in text.

Valence: The fundamental tone of the action
  - positive: Helps, cooperates, supports, befriends
  - negative: Harms, opposes, threatens, attacks

Pattern: Your own descriptive label for the action type
  - Examples: "rescues", "betrays", "shares_intel", "threatens", "defends"
  - Use specific, domain-appropriate terminology
  - No fixed list—describe what you observe

Axes: Multi-dimensional rating
  - salience: low/medium/high (how noticeable)
  - stakes: minor/moderate/major (consequences)
  - durability: momentary/temporary/persistent
  - volition: voluntary/coerced/accidental
  - goal_alignment: aligned/orthogonal/opposed
  - consequence_refs: [] (scene numbers impacted later)

Durability = "persistent" if any of:
  - Institutional pledge or order
  - Affiliation change (joining/leaving organization)
  - Death
  - Identity reveal
  - Legal verdict
  - Irreversible destruction of core asset

HARD RULES:

1. Evidence (MANDATORY): Direct quote from scene text (≤200 chars)
   - Must be substring from input
   - NO inference, NO "seems to", NO implied actions

2. Direction (CRITICAL): Every action has clear actor → target
   - pair: [actor_name, target_name] (exactly 2 names)
   - If multiple actors or targets, create separate action entries
   - If you cannot confidently identify BOTH sides, OMIT

3. ID Format (STRICT): "act_{scene:03d}_{index:04d}"
   - scene: zero-padded scene number
   - index: starts at 1, increments per action

4. Consequence_refs: List scene numbers this action will impact
   - Only if explicitly stated or strongly implied by text
   - Use [] if none

OUTPUT SCHEMA:

{
  "participants": [list of all person names in scene],
  "act_clues": [
    {
      "id": "act_005_0001",
      "scene": 5,
      "pair": ["ActorName", "TargetName"],
      "clue_type": "act",
      "evidence": "direct quote ≤200 chars",
      "actors": ["ActorName"],
      "targets": ["TargetName"],
      "valence": "positive|negative",
      "pattern": "your_descriptive_label",
      "axes": {
        "salience": "low|medium|high",
        "stakes": "minor|moderate|major",
        "durability": "momentary|temporary|persistent",
        "volition": "voluntary|coerced|accidental",
        "goal_alignment": "aligned|orthogonal|opposed",
        "consequence_refs": [7, 9]
      }
    }
  ]
}

QUALITY GUARDS:
- Keep evidence ≤200 chars (hard clip)
- Omit any action you're not fully confident about
- NO assumptions about camera work or unstated plot devices
- Names must match exactly as written in scene text
""".strip()

ACT_PROMPT_SECTION = f"## ACT CLUES\n{ACT_PROMPT_RULES}"

ACT_SYSTEM_PROMPT = (
    "You extract structured relationship ACTION clues from a single scene transcript.\n"
    "Return ONLY JSON that satisfies the provided response schema exactly.\n\n"
    f"{ACT_PROMPT_RULES}"
)


def _act_user_prompt(scene_id: int, text: str) -> str:
    return f"""SCENE_ID: {scene_id}\nTEXT:\n{text}\n\nExtract only the required action clues.""".strip()


class _ActExtractionPayload(BaseModel):
    participants: list[str] = Field(default_factory=list)
    act_clues: list[ActClueAPI] = Field(default_factory=list)

    def to_internal(self) -> tuple[list[str], list[ActClue]]:
        return self.participants, [a.to_internal() for a in self.act_clues]


class ActValidator(ClueValidator):
    def validate_semantic(self, clue: ActClue) -> ValidationResult:
        warnings: list[str] = []
        if clue.axes.stakes == "major" and not clue.axes.consequence_refs:
            warnings.append("major stakes usually reference downstream scenes")
        return ValidationResult.ok(level="semantic", warnings=warnings)

    def validate_coherence(
        self, clue: ActClue, context: Mapping[str, object] | None = None
    ) -> ValidationResult | None:
        if context is None:
            return None
        known = context.get("known_scenes", set())
        known_set = set(known) if isinstance(known, (set, list, tuple)) else set()
        missing = [ref for ref in clue.axes.consequence_refs if ref not in known_set]
        if missing:
            return ValidationResult.ok(
                level="coherence",
                warnings=[
                    "act clue consequence_refs reference unknown scenes: "
                    + ", ".join(str(m) for m in missing)
                ],
            )
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
            self._batch_size = config.batch_size
        if self._batch_size is None:
            self._batch_size = 10
        if self._client is None:
            raise ValueError("ActExtractor requires a client; none provided in config")

    def _build_inline_requests(
        self, scenes: list[dict]
    ) -> list[types.InlinedRequestDict]:
        requests: list[types.InlinedRequestDict] = []
        for item in scenes:
            sid = int(item["scene"])
            text = str(item["text"])
            requests.append(
                types.InlinedRequestDict(
                    contents=[
                        {
                            "role": "user",
                            "parts": [{"text": _act_user_prompt(sid, text)}],
                        }
                    ],
                    config=types.GenerateContentConfigDict(
                        system_instruction=ACT_SYSTEM_PROMPT,
                        response_schema=_ActExtractionPayload,
                        response_mime_type="application/json",
                    ),
                )
            )
        return requests

    def _parse_response(
        self, raw_payload: Any, scene_id: int
    ) -> tuple[list[str], list[ActClue]]:
        payload = parse_model(_ActExtractionPayload, raw_payload)
        return payload.to_internal()

    def get_prompt_section(self) -> str:
        return ACT_PROMPT_SECTION

    def get_api_model(self) -> Type[BaseModel]:
        return ActClueAPI

    def score(self, clue: ActClue) -> float:
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
    consequence_refs: list[int] = Field(default_factory=list)


class ActClue(PairClue):
    clue_type: Literal["act"] = "act"
    actors: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    valence: ValenceCore
    pattern: str
    axes: Axes

    @computed_field(return_type=str)
    @property
    def stance(self) -> str:
        return "cooperation" if self.valence == "positive" else "hostility"

    @computed_field(return_type=str)
    @property
    def subtype(self) -> str:
        return self.pattern


class ActClueAPI(BaseModel):
    id: str | None = None
    scene: int
    pair: list[str] = Field(min_length=2, max_length=2)
    clue_type: Literal["act"] = "act"
    evidence: str
    actors: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    valence: ValenceCore
    pattern: str
    axes: Axes

    @field_validator("evidence")
    @classmethod
    def _clip_evidence(cls, v: str) -> str:
        v = v.strip()
        return v if len(v) <= 200 else v[:200]

    def to_internal(self) -> ActClue:
        data = self.model_dump()
        data["pair"] = tuple(data["pair"])
        data["id"] = data.get("id") or ""
        return ActClue.model_validate(data)


__all__ = [
    "ActExtractor",
    "ActValidator",
    "ActClue",
    "ActClueAPI",
    "Axes",
    "ValenceCore",
    "act_score",
    "explode_directed",
    "bundle_same_scene",
]
