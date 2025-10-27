from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from framework.registry import ClueRegistry, ProcessorResultRegistry
from framework.result import PipelineResult, PipelineResultSnapshot


class SessionState(BaseModel):
    """Metadata describing the progress of a checkpointed pipeline session."""

    version: int = 1
    created_at: str
    last_updated: str
    scenes_hash: str
    metadata_hash: str
    extractor_progress: dict[str, list[int]] = Field(default_factory=dict)
    processor_completed: dict[str, bool] = Field(default_factory=dict)


class CheckpointSession:
    """Manage persistence for pipeline execution and resume support."""

    def __init__(self, checkpoint_dir: Path, *, resume: bool = False) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.checkpoint_dir / "session.json"
        self.result_file = self.checkpoint_dir / "result.json"

        if resume and self.state_file.exists():
            self.state = SessionState.model_validate_json(
                self.state_file.read_text(encoding="utf-8")
            )
        else:
            now = datetime.now().isoformat()
            self.state = SessionState(
                created_at=now,
                last_updated=now,
                scenes_hash="",
                metadata_hash="",
            )

    def validate_inputs(self, scenes: list, metadata: dict) -> bool:
        """Return True if inputs match the stored session signature."""

        scenes_hash = self._hash(scenes)
        metadata_hash = self._hash(metadata)

        if not self.state.scenes_hash:
            self.state.scenes_hash = scenes_hash
            self.state.metadata_hash = metadata_hash
            self._save_state()
            return True

        return (
            self.state.scenes_hash == scenes_hash
            and self.state.metadata_hash == metadata_hash
        )

    def can_resume(self) -> bool:
        """Return True when a stored PipelineResult snapshot exists."""

        return self.result_file.exists()

    def is_scene_completed(self, extractor_id: str, scene_id: int) -> bool:
        """Check whether a scene has already been processed by an extractor."""

        completed = self.state.extractor_progress.get(extractor_id, [])
        return int(scene_id) in completed

    def mark_scenes_completed(
        self, extractor_id: str, scene_ids: list[int]
    ) -> None:
        """Record completion for the provided scene ids."""

        progress = self.state.extractor_progress.setdefault(extractor_id, [])
        merged = {int(scene_id) for scene_id in progress}
        merged.update(int(scene_id) for scene_id in scene_ids)
        self.state.extractor_progress[extractor_id] = sorted(merged)
        self._save_state()

    def is_processor_completed(self, processor_id: str) -> bool:
        """Return True when processor execution has already finished."""

        return self.state.processor_completed.get(processor_id, False)

    def mark_processor_completed(self, processor_id: str) -> None:
        """Persist completion flag for a processor."""

        self.state.processor_completed[processor_id] = True
        self._save_state()

    def save_result(self, result: PipelineResult) -> None:
        """Persist the full PipelineResult snapshot to disk."""

        snapshot = result.checkpoint_state()
        self.result_file.write_text(
            snapshot.model_dump_json(indent=2), encoding="utf-8"
        )

    def load_result(
        self,
        registry: ClueRegistry,
        result_registry: ProcessorResultRegistry,
    ) -> PipelineResult:
        """Restore PipelineResult from the stored snapshot."""

        if not self.result_file.exists():
            raise FileNotFoundError(
                f"No checkpoint snapshot found at {self.result_file}"
            )

        snapshot = PipelineResultSnapshot.model_validate_json(
            self.result_file.read_text(encoding="utf-8")
        )
        result = PipelineResult()
        result.restore_state(snapshot, registry, result_registry)
        return result

    def clear(self) -> None:
        """Remove all stored checkpoint files."""

        if self.state_file.exists():
            self.state_file.unlink()
        if self.result_file.exists():
            self.result_file.unlink()

    def _save_state(self) -> None:
        """Write the session metadata to disk."""

        self.state.last_updated = datetime.now().isoformat()
        self.state_file.write_text(
            self.state.model_dump_json(indent=2), encoding="utf-8"
        )

    @staticmethod
    def _hash(value: Any) -> str:
        """Compute a stable hash for the provided value."""

        data = json.dumps(value, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


__all__ = ["CheckpointSession", "SessionState"]
