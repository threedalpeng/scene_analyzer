from __future__ import annotations

import time
from collections import defaultdict
from typing import Iterable, Literal, Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, computed_field, field_validator

from clues.base import ClueExtractor, ClueValidator
from schema import (
    Durability,
    GoalAlignment,
    PairSignal,
    Salience,
    Stakes,
    ValidationResult,
    Volition,
)
from utils import log_status, parse_model

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


ACT_SYSTEM_PROMPT = """
You extract structured relationship ACTION signals from a single scene transcript.
Return ONLY JSON that satisfies the provided response schema exactly.

{prompt_body}
""".strip().format(
    prompt_body="""
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

3. ID Format (STRICT): "act_{clue_id}_{scene:03d}_{index:04d}"
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
      "id": "act_{clue_id}_005_0001",
      "scene": 5,
      "pair": ["ActorName", "TargetName"],
      "modality": "act",
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
"""
)


STAKE_RANK = {"major": 3, "moderate": 2, "minor": 1}
SALIENCE_RANK = {"high": 3, "medium": 2, "low": 1}
DURABILITY_RANK = {"persistent": 3, "temporary": 2, "momentary": 1}


def act_score(clue: "ActClue") -> int:
    return (
        100 * STAKE_RANK.get(clue.axes.stakes, 0)
        + 10 * SALIENCE_RANK.get(clue.axes.salience, 0)
        + DURABILITY_RANK.get(clue.axes.durability, 0)
    )


def _act_user_prompt(scene_id: int, text: str) -> str:
    return f"""SCENE_ID: {scene_id}\nTEXT:\n{text}\n\nExtract only the required action clues.""".strip()


class _ActExtractionPayload(BaseModel):
    participants: list[str] = Field(default_factory=list)
    act_clues: list[ActClueAPI] = Field(default_factory=list)

    def to_internal(self) -> tuple[list[str], list[ActClue]]:
        return self.participants, [a.to_internal() for a in self.act_clues]


class ActValidator(ClueValidator):
    def validate_semantic(self, signal: ActClue) -> ValidationResult:
        warnings: list[str] = []
        if signal.axes.stakes == "major" and not signal.axes.consequence_refs:
            warnings.append("major stakes usually reference downstream scenes")
        return ValidationResult.ok(level="semantic", warnings=warnings)


class ActExtractor(ClueExtractor):
    """LLM-backed extractor for action clues."""

    def __init__(self, client: genai.Client, *, batch_size: int = 10) -> None:
        self._client = client
        self._batch_size = batch_size
        self._participants: dict[int, list[str]] = {}
        self._id_counters: defaultdict[int, int] = defaultdict(int)

    @property
    def clue_id(self) -> str:  # noqa: D401
        return "act"

    def extract(self, scene_text: str, scene_id: int) -> Sequence[ActClue]:
        return self.batch_extract([(scene_id, scene_text)])

    def batch_extract(self, items: Iterable[tuple[int, str]]) -> list[ActClue]:
        scenes = [{"scene": sid, "text": txt} for sid, txt in items]
        return self._run_batch(scenes)

    def _run_batch(self, scenes: list[dict]) -> list[ActClue]:
        outputs: list[ActClue] = []
        if not scenes:
            return outputs

        chunk = self._batch_size
        total = (len(scenes) + chunk - 1) // chunk

        for i in range(0, len(scenes), chunk):
            sub = scenes[i : i + chunk]
            batch_idx = (i // chunk) + 1
            log_status(
                f"ACT batch {batch_idx}/{total}: submitting {len(sub)} scenes to Gemini"
            )
            inlined = self._build_inline_requests(sub)
            job = self._client.batches.create(
                model="gemini-2.5-flash",
                src=types.BatchJobSourceDict(inlined_requests=inlined),
                config=types.CreateBatchJobConfigDict(
                    display_name=f"act-{i // chunk:03d}",
                ),
            )

            assert job.name is not None
            done_states = {
                "JOB_STATE_SUCCEEDED",
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
                "JOB_STATE_EXPIRED",
            }
            last_state: str | None = None
            while True:
                bj = self._client.batches.get(name=job.name)
                assert bj.state is not None
                state_name = bj.state.name
                if state_name != last_state:
                    log_status(f"ACT batch {batch_idx}/{total}: {state_name.lower()}")
                    last_state = state_name
                if state_name in done_states:
                    if state_name != "JOB_STATE_SUCCEEDED":
                        raise RuntimeError(f"ACT batch failed: {state_name} {bj.error}")
                    break
                time.sleep(3)

            assert bj.dest is not None and bj.dest.inlined_responses is not None
            for idx, resp in enumerate(bj.dest.inlined_responses, start=1):
                if resp.error:
                    log_status(
                        f"ACT batch {batch_idx}/{total}: inline {idx} error -> {resp.error}"
                    )
                    continue
                parsed = (
                    getattr(resp.response, "parsed", None) if resp.response else None
                )
                raw_payload = parsed or getattr(resp.response, "text", None)
                if raw_payload is None:
                    log_status(
                        f"ACT batch {batch_idx}/{total}: inline {idx} empty response"
                    )
                    continue

                try:
                    payload = parse_model(_ActExtractionPayload, raw_payload)
                except ValidationError as err:
                    log_status(
                        f"ACT batch {batch_idx}/{total}: inline {idx} parse error -> {err}"
                    )
                    continue

                scene_id = int(sub[idx - 1]["scene"])
                participants, clues = payload.to_internal()
                clues = self._assign_ids(scene_id, clues)
                self._participants[scene_id] = participants
                outputs.extend(clues)
        return outputs

    def _assign_ids(self, scene_id: int, clues: list[ActClue]) -> list[ActClue]:
        assigned: list[ActClue] = []
        for clue in clues:
            self._id_counters[scene_id] += 1
            new_id = f"act_auto_{scene_id:03d}_{self._id_counters[scene_id]:04d}"
            assigned.append(clue.model_copy(update={"id": new_id}))
        return assigned

    def _build_inline_requests(
        self,
        scenes: list[dict],
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

    def score(self, clue: ActClue) -> float:
        return float(act_score(clue))

    def validator(self) -> ClueValidator:
        return ActValidator()

    def participants(self) -> dict[int, list[str]]:
        return self._participants


ValenceCore = Literal["positive", "negative"]


class Axes(BaseModel):
    salience: Salience
    stakes: Stakes
    durability: Durability
    volition: Volition
    goal_alignment: GoalAlignment
    consequence_refs: list[int] = Field(default_factory=list)


class ActClue(PairSignal):
    modality: Literal["act"] = "act"
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
    modality: Literal["act"] = "act"
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
]
