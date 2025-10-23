from __future__ import annotations

import time
from collections import defaultdict
from typing import Iterable, Literal, Mapping, Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, field_validator

from clues.base import ClueExtractor, ClueValidator
from schema import BaseClue, ValidationResult
from utils import log_status, parse_model


TEMPORAL_SYSTEM_PROMPT = """
You extract TEMPORAL RELATIONSHIPS between scenes from scene text.
Return ONLY JSON that satisfies the provided response schema exactly.

PURPOSE: Help reconstruct chronological order (Fabula) from presentation order (Syuzhet).

{rules}
""".strip().format(
    rules="""
CORE CONCEPTS:

References_scenes: Which past scenes does THIS scene reference or continue?
Time_offset: Approximate time difference from narrative present (in days)
Is_flashback: True if the scene is presented out of chronological order.

HARD RULES:
- Evidence (≤200 chars) must be direct quote indicating the temporal relationship.
- Use ONLY explicit textual evidence: temporal markers, dialogue references, SDH cues.
- ID Format: "temporal_{scene:03d}_{index:04d}".

OUTPUT SCHEMA:
{
  "temporal_clues": [
    {
      "id": "temporal_005_0001",
      "scene": 5,
      "clue_type": "temporal",
      "references_scenes": [1, 2],
      "time_offset": -1095,
      "is_flashback": true,
      "evidence": "direct quote"
    }
  ]
}
"""
)


def _temporal_user_prompt(scene_id: int, text: str) -> str:
    return f"""SCENE_ID: {scene_id}\nTEXT:\n{text}\n\nExtract explicit temporal clues only.""".strip()


class _TemporalExtractionPayload(BaseModel):
    temporal_clues: list[TemporalClueAPI] = Field(default_factory=list)

    def to_internal(self) -> list[TemporalClue]:
        return [c.to_internal() for c in self.temporal_clues]


class TemporalValidator(ClueValidator):
    def validate_semantic(self, clue: BaseClue) -> ValidationResult:
        _ = clue
        return ValidationResult.ok(level="semantic")

    def validate_coherence(
        self, clue: BaseClue, context: Mapping[str, object] | None = None
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


class TemporalExtractor(ClueExtractor):
    def __init__(self, client: genai.Client, *, batch_size: int = 10) -> None:
        self._client = client
        self._batch_size = batch_size
        self._id_counters: defaultdict[int, int] = defaultdict(int)

    @property
    def clue_type(self) -> str:  # noqa: D401
        return "temporal"

    def extract(self, scene_text: str, scene_id: int) -> Sequence[TemporalClue]:
        return self.batch_extract([(scene_id, scene_text)])

    def batch_extract(self, items: Iterable[tuple[int, str]]) -> list[TemporalClue]:
        scenes = [{"scene": sid, "text": txt} for sid, txt in items]
        return self._run_batch(scenes)

    def _run_batch(self, scenes: list[dict]) -> list[TemporalClue]:
        outputs: list[TemporalClue] = []
        if not scenes:
            return outputs

        chunk = self._batch_size
        total = (len(scenes) + chunk - 1) // chunk

        for i in range(0, len(scenes), chunk):
            sub = scenes[i : i + chunk]
            batch_idx = (i // chunk) + 1
            log_status(
                f"TEMPORAL batch {batch_idx}/{total}: submitting {len(sub)} scenes"
            )
            inlined = self._build_inline_requests(sub)
            job = self._client.batches.create(
                model="gemini-2.5-flash",
                src=types.BatchJobSourceDict(inlined_requests=inlined),
                config=types.CreateBatchJobConfigDict(
                    display_name=f"temporal-{i // chunk:03d}",
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
                        f"TEMPORAL batch {batch_idx}/{total}: {state_name.lower()}"
                    )
                    last_state = state_name
                if state_name in done_states:
                    if state_name != "JOB_STATE_SUCCEEDED":
                        raise RuntimeError(
                            f"TEMPORAL batch failed: {state_name} {bj.error}"
                        )
                    break
                time.sleep(3)

            assert bj.dest is not None and bj.dest.inlined_responses is not None
            for idx, resp in enumerate(bj.dest.inlined_responses, start=1):
                if resp.error:
                    log_status(
                        f"TEMPORAL batch {batch_idx}/{total}: inline {idx} error -> {resp.error}"
                    )
                    continue
                parsed = getattr(resp.response, "parsed", None) if resp.response else None
                raw_payload = parsed or getattr(resp.response, "text", None)
                if raw_payload is None:
                    log_status(
                        f"TEMPORAL batch {batch_idx}/{total}: inline {idx} empty response"
                    )
                    continue

                try:
                    payload = parse_model(_TemporalExtractionPayload, raw_payload)
                except ValidationError as err:
                    log_status(
                        f"TEMPORAL batch {batch_idx}/{total}: inline {idx} parse error -> {err}"
                    )
                    continue

                scene_id = int(sub[idx - 1]["scene"])
                clues = self._assign_ids(scene_id, payload.to_internal())
                outputs.extend(clues)
        return outputs

    def _assign_ids(self, scene_id: int, clues: list[TemporalClue]) -> list[TemporalClue]:
        assigned: list[TemporalClue] = []
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
                            "parts": [{"text": _temporal_user_prompt(sid, text)}],
                        }
                    ],
                    config=types.GenerateContentConfigDict(
                        system_instruction=TEMPORAL_SYSTEM_PROMPT,
                        response_schema=_TemporalExtractionPayload,
                        response_mime_type="application/json",
                    ),
                )
            )
        return requests

    def score(self, clue: TemporalClue) -> float:
        _ = clue
        return 0.0

    def validator(self) -> ClueValidator:
        return TemporalValidator()


class TemporalClue(BaseClue):
    clue_type: Literal["temporal"] = "temporal"
    references_scenes: list[int] = Field(default_factory=list)
    time_offset: int | None = None
    is_flashback: bool


class TemporalClueAPI(BaseModel):
    id: str | None = None
    scene: int
    clue_type: Literal["temporal"] = "temporal"
    evidence: str
    references_scenes: list[int] = Field(default_factory=list)
    time_offset: int | None = None
    is_flashback: bool

    @field_validator("evidence")
    @classmethod
    def _clip_evidence(cls, v: str) -> str:
        v = v.strip()
        return v if len(v) <= 200 else v[:200]

    def to_internal(self) -> TemporalClue:
        data = self.model_dump()
        data["id"] = data.get("id") or ""
        return TemporalClue.model_validate(data)


__all__ = ["TemporalExtractor", "TemporalClue", "TemporalClueAPI"]
