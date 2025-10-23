from __future__ import annotations

import time
from collections import defaultdict
from typing import Iterable, Literal, Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, field_validator

from clues.base import ClueExtractor, ClueValidator
from schema import PairClue, ValidationResult
from utils import log_status, parse_model


TOM_SYSTEM_PROMPT = """
You extract structured THEORY OF MIND clues from a single scene transcript.
Return ONLY JSON that satisfies the provided response schema exactly.

CORE CONCEPTS:

Theory of Mind: Mental states (beliefs, feelings, intentions, desires) that 
one character has ABOUT another character. Must be explicitly signaled.

ToM Kinds:
  - BelievesAbout: What A thinks about B
  - FeelsTowards: What emotional state A has toward B
  - IntendsTo: What A plans to do regarding B
  - DesiresFor: What A wants from/for B

Claim: Short, literal statement of the mental state
  - Example: "Alice believes Bob stole the documents"
  - NOT: "Alice seems suspicious of Bob"

HARD RULES:

1. Evidence (MANDATORY): Direct quote from scene (≤200 chars)
   - Dialogue expressing belief or intention
   - Narration describing mental state
  - Described behavior that explicitly indicates the state

2. Explicit Signal Required:
   - Must be stated in dialogue, narration, or SDH cues
   - NO inference from context alone
   - If target is ambiguous, OMIT the entry

3. Direction (pair): [thinker_name, target_name]
   - Thinker: Who has the mental state
   - Target: Who the mental state is about
   - Exactly 2 names required

4. ID Format: "tom_{scene:03d}_{index:04d}"

OUTPUT SCHEMA:

{
  "participants": [list of all person names],
  "tom_clues": [
    {
      "id": "tom_005_0001",
      "scene": 5,
      "pair": ["Thinker", "Target"],
      "clue_type": "tom",
      "evidence": "direct quote ≤200 chars",
      "kind": "BelievesAbout|FeelsTowards|IntendsTo|DesiresFor",
      "claim": "short literal statement"
    }
  ]
}

QUALITY GUARDS:
- Keep evidence ≤200 chars
- Omit any clue you're not fully confident about
- Names must match the scene text exactly
"""


def _tom_user_prompt(scene_id: int, text: str) -> str:
    return f"""SCENE_ID: {scene_id}\nTEXT:\n{text}\n\nExtract only the required theory-of-mind clues.""".strip()


class _ToMExtractionPayload(BaseModel):
    participants: list[str] = Field(default_factory=list)
    tom_clues: list[ToMClueAPI] = Field(default_factory=list)

    def to_internal(self) -> tuple[list[str], list[ToMClue]]:
        return self.participants, [c.to_internal() for c in self.tom_clues]


class ToMValidator(ClueValidator):
    def validate_semantic(self, clue: ToMClue) -> ValidationResult:
        if not clue.claim:
            return ValidationResult.fail(
                level="semantic", errors=["claim must be non-empty"]
            )
        return ValidationResult.ok(level="semantic")


class ToMExtractor(ClueExtractor):
    def __init__(self, client: genai.Client, *, batch_size: int = 10) -> None:
        self._client = client
        self._batch_size = batch_size
        self._participants: dict[int, list[str]] = {}
        self._id_counters: defaultdict[int, int] = defaultdict(int)

    @property
    def clue_type(self) -> str:  # noqa: D401
        return "tom"

    def extract(self, scene_text: str, scene_id: int) -> Sequence[ToMClue]:
        return self.batch_extract([(scene_id, scene_text)])

    def batch_extract(self, items: Iterable[tuple[int, str]]) -> list[ToMClue]:
        scenes = [{"scene": sid, "text": txt} for sid, txt in items]
        return self._run_batch(scenes)

    def _run_batch(self, scenes: list[dict]) -> list[ToMClue]:
        outputs: list[ToMClue] = []
        if not scenes:
            return outputs

        chunk = self._batch_size
        total = (len(scenes) + chunk - 1) // chunk

        for i in range(0, len(scenes), chunk):
            sub = scenes[i : i + chunk]
            batch_idx = (i // chunk) + 1
            log_status(
                f"TOM batch {batch_idx}/{total}: submitting {len(sub)} scenes to Gemini"
            )
            inlined = self._build_inline_requests(sub)
            job = self._client.batches.create(
                model="gemini-2.5-flash",
                src=types.BatchJobSourceDict(inlined_requests=inlined),
                config=types.CreateBatchJobConfigDict(
                    display_name=f"tom-{i // chunk:03d}",
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
                    log_status(
                        f"TOM batch {batch_idx}/{total}: {state_name.lower()}"
                    )
                    last_state = state_name
                if state_name in done_states:
                    if state_name != "JOB_STATE_SUCCEEDED":
                        raise RuntimeError(f"TOM batch failed: {state_name} {bj.error}")
                    break
                time.sleep(3)

            assert bj.dest is not None and bj.dest.inlined_responses is not None
            for idx, resp in enumerate(bj.dest.inlined_responses, start=1):
                if resp.error:
                    log_status(
                        f"TOM batch {batch_idx}/{total}: inline {idx} error -> {resp.error}"
                    )
                    continue
                parsed = getattr(resp.response, "parsed", None) if resp.response else None
                raw_payload = parsed or getattr(resp.response, "text", None)
                if raw_payload is None:
                    log_status(
                        f"TOM batch {batch_idx}/{total}: inline {idx} empty response"
                    )
                    continue

                try:
                    payload = parse_model(_ToMExtractionPayload, raw_payload)
                except ValidationError as err:
                    log_status(
                        f"TOM batch {batch_idx}/{total}: inline {idx} parse error -> {err}"
                    )
                    continue

                scene_id = int(sub[idx - 1]["scene"])
                participants, clues = payload.to_internal()
                clues = self._assign_ids(scene_id, clues)
                self._participants[scene_id] = participants
                outputs.extend(clues)
        return outputs

    def _assign_ids(self, scene_id: int, clues: list[ToMClue]) -> list[ToMClue]:
        assigned: list[ToMClue] = []
        for clue in clues:
            self._id_counters[scene_id] += 1
            new_id = f"{self.clue_type}_{scene_id:03d}_{self._id_counters[scene_id]:04d}"
            assigned.append(clue.model_copy(update={"id": new_id}))
        return assigned

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
                            "parts": [{"text": _tom_user_prompt(sid, text)}],
                        }
                    ],
                    config=types.GenerateContentConfigDict(
                        system_instruction=TOM_SYSTEM_PROMPT,
                        response_schema=_ToMExtractionPayload,
                        response_mime_type="application/json",
                    ),
                )
            )
        return requests

    def validator(self) -> ClueValidator:
        return ToMValidator()

    def participants(self) -> dict[int, list[str]]:
        return self._participants

    def score(self, clue: ToMClue) -> float:
        _ = clue
        return 0.0


ToMKind = Literal["BelievesAbout", "FeelsTowards", "IntendsTo", "DesiresFor"]


class ToMClue(PairClue):
    clue_type: Literal["tom"] = "tom"
    kind: ToMKind
    claim: str


class ToMClueAPI(BaseModel):
    id: str | None = None
    scene: int
    pair: list[str] = Field(min_length=2, max_length=2)
    clue_type: Literal["tom"] = "tom"
    evidence: str
    kind: ToMKind
    claim: str

    @field_validator("evidence")
    @classmethod
    def _clip_evidence(cls, v: str) -> str:
        v = v.strip()
        return v if len(v) <= 200 else v[:200]

    def to_internal(self) -> ToMClue:
        data = self.model_dump()
        data["pair"] = tuple(data["pair"])
        data["id"] = data.get("id") or ""
        return ToMClue.model_validate(data)


__all__ = ["ToMExtractor", "ToMValidator", "ToMClue", "ToMClueAPI", "ToMKind"]
