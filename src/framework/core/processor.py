"""Abstract base class for pipeline processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type

from pydantic import BaseModel

from framework.core.result import PipelineResult

if TYPE_CHECKING:
    from framework.core.pipeline import PipelineConfig


class Processor(ABC):
    """
    Base class for pipeline processors.

    Processors operate on the accumulated :class:`PipelineResult` rather than
    individual segments. Subclasses typically store configuration in
    :meth:`configure` and transform results in :meth:`__call__`.
    """

    @abstractmethod
    def configure(self, config: PipelineConfig) -> None:
        """Update processor settings using pipeline configuration."""

    @abstractmethod
    def __call__(self, result: PipelineResult) -> BaseModel | None:
        """Process the accumulated pipeline result and optionally return output."""

    @property
    def checkpoint_id(self) -> str:
        """Return a stable identifier used for checkpoint bookkeeping."""
        cls = self.__class__
        return f"{cls.__module__}.{cls.__qualname__}"

    @property
    def result_type(self) -> Type[BaseModel] | None:
        """Return the BaseModel type produced by this processor (if any)."""
        return None
