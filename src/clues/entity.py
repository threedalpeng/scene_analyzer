from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, Type

from google.genai import types
from pydantic import BaseModel, Field, field_validator

from framework.base import BatchExtractor, ClueValidator
from schema import BaseClue, ValidationResult
from utils import parse_model

if TYPE_CHECKING:
    from framework.pipeline import PipelineConfig


ENTITY_PROMPT_RULES = """
PURPOSE: Help resolve multiple names referring to the same person.

CORE CONCEPTS:
- name: primary name used for the character in this scene
- aliases_in_scene: other names/titles explicitly referring to the same person

HARD RULES:
1. Evidence must be a direct quote (≤200 chars) showing the alias relationship.
2. Extract ONLY if both names appear in the scene and clearly refer to the same person.
3. ID format: "entity_{scene:03d}_{index:04d}".

OUTPUT SCHEMA:
{
  "participants": [list of person names],
  "entity_clues": [
    {
      "id": "entity_005_0001",
      "scene": 5,
      "clue_type": "entity",
      "name": "Primary Name",
      "aliases_in_scene": ["Alias"],
      "evidence": "direct quote"
    }
  ]
}
""".strip()

ENTITY_PROMPT_SECTION = f"## ENTITY CLUES\n{ENTITY_PROMPT_RULES}"

ENTITY_SYSTEM_PROMPT = (
    "You extract CHARACTER IDENTITIES and ALIAS relationships from scene text.\n"
    "Return ONLY JSON that satisfies the provided response schema exactly.\n\n"
    f"{ENTITY_PROMPT_RULES}"
)


def _entity_user_prompt(scene_id: int, text: str) -> str:
    return f"""SCENE_ID: {scene_id}\nTEXT:\n{text}\n\nExtract only explicit alias relationships.""".strip()


class _EntityExtractionPayload(BaseModel):
    participants: list[str] = Field(default_factory=list)
    entity_clues: list[EntityClueAPI] = Field(default_factory=list)

    def to_internal(self) -> tuple[list[str], list[EntityClue]]:
        return self.participants, [c.to_internal() for c in self.entity_clues]


class EntityValidator(ClueValidator):
    def validate_semantic(self, clue: EntityClue) -> ValidationResult:
        if not clue.aliases_in_scene:
            return ValidationResult.fail(
                level="semantic", errors=["alias list must not be empty"]
            )
        return ValidationResult.ok(level="semantic")

    def validate_coherence(
        self, clue: EntityClue, context: Mapping[str, object] | None = None
    ) -> ValidationResult | None:
        if clue.aliases_in_scene:
            return None
        return ValidationResult.ok(
            level="coherence",
            warnings=["entity clue should list aliases if present"],
        )


class EntityExtractor(BatchExtractor):
    _clue_slug = "entity"

    @property
    def clue_type(self) -> type["EntityClue"]:  # noqa: D401
        return EntityClue

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
                "EntityExtractor requires a client; none provided in config"
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

    def _parse_response(
        self, raw_payload: Any, scene_id: int
    ) -> tuple[list[str], list[EntityClue]]:
        payload = parse_model(_EntityExtractionPayload, raw_payload)
        return payload.to_internal()

    def get_prompt_section(self) -> str:
        return ENTITY_PROMPT_SECTION

    def get_api_model(self) -> Type[BaseModel]:
        return EntityClueAPI

    def score(self, clue: EntityClue) -> float:
        _ = clue
        return 0.0

    def validator(self) -> ClueValidator:
        return EntityValidator()


class EntityClue(BaseClue):
    clue_type: Literal["entity"] = "entity"
    name: str
    aliases_in_scene: list[str] = Field(default_factory=list)


class EntityClueAPI(BaseModel):
    id: str | None = None
    scene: int
    clue_type: Literal["entity"] = "entity"
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
