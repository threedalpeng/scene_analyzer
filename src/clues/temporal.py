from __future__ import annotations

from typing import Any, Literal, Mapping, TYPE_CHECKING, Type

from google.genai import types
from pydantic import BaseModel, Field, field_validator

from framework.base import BatchExtractor, ClueValidator
from schema import BaseClue, ValidationResult
from utils import parse_model

if TYPE_CHECKING:
    from framework.pipeline import PipelineConfig


TEMPORAL_PROMPT_RULES = """
PURPOSE: Help reconstruct chronological order (Fabula) from presentation order (Syuzhet).

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
""".strip()

TEMPORAL_PROMPT_SECTION = f"## TEMPORAL CLUES\n{TEMPORAL_PROMPT_RULES}"

TEMPORAL_SYSTEM_PROMPT = (
    "You extract TEMPORAL RELATIONSHIPS between scenes from scene text.\n"
    "Return ONLY JSON that satisfies the provided response schema exactly.\n\n"
    f"{TEMPORAL_PROMPT_RULES}"
)


def _temporal_user_prompt(scene_id: int, text: str) -> str:
    return f"""SCENE_ID: {scene_id}\nTEXT:\n{text}\n\nExtract explicit temporal clues only.""".strip()


class _TemporalExtractionPayload(BaseModel):
    temporal_clues: list[TemporalClueAPI] = Field(default_factory=list)

    def to_internal(self) -> tuple[list[str], list[TemporalClue]]:
        return [], [c.to_internal() for c in self.temporal_clues]


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


class TemporalExtractor(BatchExtractor):
    _clue_slug = "temporal"

    @property
    def clue_type(self) -> type["TemporalClue"]:  # noqa: D401
        return TemporalClue

    def configure(self, config: "PipelineConfig") -> None:
        super().configure(config)
        if self._client is None:
            self._client = config.client
        if self._batch_size is None:
            self._batch_size = config.batch_size
        if self._batch_size is None:
            self._batch_size = 50
        if self._client is None:
            raise ValueError(
                "TemporalExtractor requires a client; none provided in config"
            )

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

    def _parse_response(
        self, raw_payload: Any, scene_id: int
    ) -> tuple[list[str], list[TemporalClue]]:
        payload = parse_model(_TemporalExtractionPayload, raw_payload)
        return payload.to_internal()

    def get_prompt_section(self) -> str:
        return TEMPORAL_PROMPT_SECTION

    def get_api_model(self) -> Type[BaseModel]:
        return TemporalClueAPI

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
