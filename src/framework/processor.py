"""Protocol definition for pipeline processors."""

from __future__ import annotations

from typing import Protocol, Type

from pydantic import BaseModel

from framework.pipeline import PipelineConfig
from framework.result import PipelineResult


class Processor(Protocol):
    """Simplified processor contract for the pipeline."""

    def configure(self, config: PipelineConfig) -> None:
        """Update processor settings using pipeline configuration."""
        ...

    def __call__(self, result: PipelineResult) -> BaseModel | None:
        """Process the accumulated pipeline result and optionally return output."""
        ...

    def checkpoint_id(self) -> str:
        """Return a stable identifier used for checkpoint bookkeeping."""
        ...

    @property
    def result_type(self) -> Type[BaseModel] | None:
        """Return the BaseModel type produced by this processor (if any)."""
        ...
