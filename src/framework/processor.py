"""Protocol definition for pipeline processors."""

from __future__ import annotations

from typing import Mapping, Optional, Protocol, TypeVar

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

    def checkpoint_state(
        self, result: PipelineResult, output: ProcessorResultT | None
    ) -> Mapping[str, object] | None:
        """
        Optional hook to serialize processor output for checkpoints.

        Return JSON-serializable payload or None to opt out.
        """
        _ = result, output
        return None

    def restore_from_checkpoint(
        self, payload: Mapping[str, object], result: PipelineResult
    ) -> Optional[ProcessorResultT]:
        """
        Optional hook to reconstruct processor output during resume.

        Should return the restored output (or None) and may mutate PipelineResult.
        """
        _ = payload, result
        return None
