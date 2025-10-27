"""Framework core for the narrative relationship analysis pipeline."""

from .checkpoint_session import CheckpointSession
from .pipeline import Pipeline, PipelineConfig
from .processor import Processor
from .result import PipelineResult, PipelineResultSnapshot
from .validation import ValidationPipeline
from .registry import ProcessorResultRegistry

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineResultSnapshot",
    "ValidationPipeline",
    "Processor",
    "CheckpointSession",
    "ProcessorResultRegistry",
]
