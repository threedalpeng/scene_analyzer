from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING, Type

from google.genai import types
from pydantic import BaseModel, Field, field_validator

from framework.base import ClueValidator
from framework.batch import BatchExtractor
from schema import PairClue, ValidationResult
from utils import parse_model

if TYPE_CHECKING:
    from framework.pipeline import PipelineConfig


TOM_PROMPT_RULES = """
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
""".strip()

TOM_PROMPT_SECTION = f"## TOM CLUES\n{TOM_PROMPT_RULES}"

TOM_SYSTEM_PROMPT = (
    "You extract structured THEORY OF MIND clues from a single scene transcript.\n"
    "Return ONLY JSON that satisfies the provided response schema exactly.\n\n"
    f"{TOM_PROMPT_RULES}"
)


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


class ToMExtractor(BatchExtractor):
    _clue_slug = "tom"

    @property
    def clue_type(self) -> type["ToMClue"]:  # noqa: D401
        return ToMClue

    def configure(self, config: "PipelineConfig") -> None:
        super().configure(config)
        if self._client is None:
            self._client = config.client
        if self._batch_size is None:
            self._batch_size = config.batch_size
        if self._batch_size is None:
            self._batch_size = 50
        if self._client is None:
            raise ValueError("ToMExtractor requires a client; none provided in config")

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

    def _parse_response(
        self, raw_payload: Any, scene_id: int
    ) -> tuple[list[str], list[ToMClue]]:
        payload = parse_model(_ToMExtractionPayload, raw_payload)
        return payload.to_internal()

    def get_prompt_section(self) -> str:
        return TOM_PROMPT_SECTION

    def get_api_model(self) -> Type[BaseModel]:
        return ToMClueAPI

    def validator(self) -> ClueValidator:
        return ToMValidator()

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
