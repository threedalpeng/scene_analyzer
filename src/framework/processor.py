"""Protocol definition for pipeline processors."""

from __future__ import annotations

from typing import Optional, Protocol, TypeVar

from framework.pipeline import PipelineConfig
from framework.result import PipelineResult

ProcessorResultT = TypeVar("ProcessorResultT", covariant=True)


class Processor(Protocol[ProcessorResultT]):
    """
    Protocol for pipeline processors.

    All processors must:
    1. Accept pipeline configuration via `configure`.
    2. Process `PipelineResult` via `__call__`, returning a result.
    """

    def configure(self, config: PipelineConfig) -> None:
        """Update processor settings using pipeline configuration."""

    def __call__(self, result: PipelineResult) -> Optional[ProcessorResultT]:
        """Process the accumulated pipeline result and optionally return output."""
