"""Framework core for the narrative relationship analysis pipeline."""

from .pipeline import Pipeline, PipelineConfig
from .processor import Processor
from .result import PipelineResult
from .validation import ValidationPipeline

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "ValidationPipeline",
    "Processor",
]
