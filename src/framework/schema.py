from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ValidationResult(BaseModel):
    passed: bool
    level: Literal["structural", "semantic", "coherence"]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def ok(
        cls,
        level: Literal["structural", "semantic", "coherence"] = "structural",
        warnings: list[str] | None = None,
    ) -> "ValidationResult":
        return cls(passed=True, level=level, warnings=warnings or [])

    @classmethod
    def fail(
        cls,
        level: Literal["structural", "semantic", "coherence"],
        errors: list[str],
        warnings: list[str] | None = None,
    ) -> "ValidationResult":
        return cls(passed=False, level=level, errors=errors, warnings=warnings or [])


class BaseClue(BaseModel):
    id: str
    segment: int
    clue_type: str
    evidence: str
    references: list[str] = Field(default_factory=list)
    referenced_segments: list[int] = Field(default_factory=list)

    @field_validator("evidence")
    @classmethod
    def _clip_evidence(cls, v: str) -> str:
        v = v.strip()
        return v if len(v) <= 200 else v[:200]
