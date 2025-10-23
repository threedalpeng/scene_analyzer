from __future__ import annotations

import time
from collections import defaultdict
from typing import Iterable, Literal, Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, field_validator

from clues.base import ClueExtractor, ClueValidator
from schema import BaseSignal, ValidationResult
from utils import log_status, parse_model


ENTITY_SYSTEM_PROMPT = """
You extract CHARACTER IDENTITIES and ALIAS relationships from scene text.
Return ONLY JSON that satisfies the provided response schema exactly.

PURPOSE: Help resolve multiple names referring to the same person.

CORE CONCEPTS:
- name: primary name used for the character in this scene
- aliases_in_scene: other names/titles explicitly referring to the same person

HARD RULES:
1. Evidence must be a direct quote (≤200 chars) showing the alias relationship.
2. Extract ONLY if both names appear in the scene and clearly refer to the same person.
3. ID format: "entity_{clue_id}_{scene:03d}_{index:04d}".

OUTPUT SCHEMA:
{
  "participants": [list of person names],
  "entity_clues": [
    {
      "id": "entity_{clue_id}_005_0001",
      "scene": 5,
      "modality": "entity",
      "name": "Primary Name",
      "aliases_in_scene": ["Alias"],
      "evidence": "direct quote"
    }
  ]
}
"""


def _entity_user_prompt(scene_id: int, text: str) -> str:
    return f"""SCENE_ID: {scene_id}\nTEXT:\n{text}\n\nExtract only explicit alias relationships.""".strip()


class _EntityExtractionPayload(BaseModel):
    participants: list[str] = Field(default_factory=list)
    entity_clues: list[EntityClueAPI] = Field(default_factory=list)

    def to_internal(self) -> tuple[list[str], list[EntityClue]]:
        return self.participants, [c.to_internal() for c in self.entity_clues]


class EntityValidator(ClueValidator):
    def validate_semantic(self, signal: EntityClue) -> ValidationResult:
        if not signal.aliases_in_scene:
            return ValidationResult.fail(
                level="semantic", errors=["alias list must not be empty"]
            )
        return ValidationResult.ok(level="semantic")


class EntityExtractor(ClueExtractor):
    def __init__(self, client: genai.Client, *, batch_size: int = 10) -> None:
        self._client = client
        self._batch_size = batch_size
        self._participants: dict[int, list[str]] = {}
        self._id_counters: defaultdict[int, int] = defaultdict(int)

    @property
    def clue_id(self) -> str:  # noqa: D401
        return "entity"

    def extract(self, scene_text: str, scene_id: int) -> Sequence[EntityClue]:
        return self.batch_extract([(scene_id, scene_text)])

    def batch_extract(self, items: Iterable[tuple[int, str]]) -> list[EntityClue]:
        scenes = [{"scene": sid, "text": txt} for sid, txt in items]
        return self._run_batch(scenes)

    def _run_batch(self, scenes: list[dict]) -> list[EntityClue]:
        outputs: list[EntityClue] = []
        if not scenes:
            return outputs

        chunk = self._batch_size
        total = (len(scenes) + chunk - 1) // chunk

        for i in range(0, len(scenes), chunk):
            sub = scenes[i : i + chunk]
            batch_idx = (i // chunk) + 1
            log_status(
                f"ENTITY batch {batch_idx}/{total}: submitting {len(sub)} scenes"
            )
            inlined = self._build_inline_requests(sub)
            job = self._client.batches.create(
                model="gemini-2.5-flash",
                src=types.BatchJobSourceDict(inlined_requests=inlined),
                config=types.CreateBatchJobConfigDict(
                    display_name=f"entity-{i // chunk:03d}",
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
                        f"ENTITY batch {batch_idx}/{total}: {state_name.lower()}"
                    )
                    last_state = state_name
                if state_name in done_states:
                    if state_name != "JOB_STATE_SUCCEEDED":
                        raise RuntimeError(
                            f"ENTITY batch failed: {state_name} {bj.error}"
                        )
                    break
                time.sleep(3)

            assert bj.dest is not None and bj.dest.inlined_responses is not None
            for idx, resp in enumerate(bj.dest.inlined_responses, start=1):
                if resp.error:
                    log_status(
                        f"ENTITY batch {batch_idx}/{total}: inline {idx} error -> {resp.error}"
                    )
                    continue
                parsed = getattr(resp.response, "parsed", None) if resp.response else None
                raw_payload = parsed or getattr(resp.response, "text", None)
                if raw_payload is None:
                    log_status(
                        f"ENTITY batch {batch_idx}/{total}: inline {idx} empty response"
                    )
                    continue

                try:
                    payload = parse_model(_EntityExtractionPayload, raw_payload)
                except ValidationError as err:
                    log_status(
                        f"ENTITY batch {batch_idx}/{total}: inline {idx} parse error -> {err}"
                    )
                    continue

                scene_id = int(sub[idx - 1]["scene"])
                participants, clues = payload.to_internal()
                clues = self._assign_ids(scene_id, clues)
                self._participants[scene_id] = participants
                outputs.extend(clues)
        return outputs

    def _assign_ids(self, scene_id: int, clues: list[EntityClue]) -> list[EntityClue]:
        assigned: list[EntityClue] = []
        for clue in clues:
            self._id_counters[scene_id] += 1
            new_id = f"entity_auto_{scene_id:03d}_{self._id_counters[scene_id]:04d}"
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
                            "parts": [{"text": _entity_user_prompt(sid, text)}],
                        }
                    ],
                    config=types.GenerateContentConfigDict(
                        system_instruction=ENTITY_SYSTEM_PROMPT,
                        response_schema=_EntityExtractionPayload,
                        response_mime_type="application/json",
                    ),
                )
            )
        return requests

    def validator(self) -> ClueValidator:
        return EntityValidator()

    def participants(self) -> dict[int, list[str]]:
        return self._participants

    def score(self, clue: EntityClue) -> float:
        _ = clue
        return 0.0


class EntityClue(BaseSignal):
    modality: Literal["entity"] = "entity"
    name: str
    aliases_in_scene: list[str] = Field(default_factory=list)


class EntityClueAPI(BaseModel):
    id: str | None = None
    scene: int
    modality: Literal["entity"] = "entity"
    evidence: str
    name: str
    aliases_in_scene: list[str] = Field(default_factory=list)

    @field_validator("evidence")
    @classmethod
    def _clip_evidence(cls, v: str) -> str:
        v = v.strip()
        return v if len(v) <= 200 else v[:200]

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
