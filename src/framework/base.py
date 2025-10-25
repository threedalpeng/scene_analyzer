from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
    Type,
    TypeVar,
)

from google.genai import types
from pydantic import ValidationError

from schema import BaseClue, ValidationResult
from utils import log_status

if TYPE_CHECKING:
    from framework.pipeline import PipelineConfig


ClueT = TypeVar("ClueT", bound=BaseClue)


class ClueValidator(ABC):
    """Domain-specific validation hooks for a clue extractor."""

    @abstractmethod
    def validate_semantic(self, clue: BaseClue) -> ValidationResult:
        """Validate semantic rules for a single clue."""

    def validate_coherence(
        self, clue: BaseClue, context: Mapping[str, Any] | None = None
    ) -> ValidationResult | None:
        """Optional cross-clue validation; return None to skip."""

        _ = clue, context
        return None


class ClueExtractor(Generic[ClueT], ABC):
    """Interface for pluggable clue extractors."""

    def __init__(self) -> None:
        self._configured = False

    def configure(self, config: "PipelineConfig") -> None:
        """Inject shared pipeline configuration before extraction."""

        self._configured = True

    @property
    @abstractmethod
    def clue_type(self) -> Type[ClueT]:
        """Concrete BaseClue subtype produced by this extractor."""

    @abstractmethod
    def extract(self, scene_text: str, scene_id: int) -> Sequence[ClueT]:
        """Extract clues from a single scene."""

    def batch_extract(self, items: Iterable[tuple[int, str]]) -> Sequence[ClueT]:
        """Optional batch extraction hook; defaults to sequential extract calls."""

        outputs: list[ClueT] = []
        for scene_id, text in items:
            outputs.extend(self.extract(text, scene_id))
        return outputs

    def score(self, clue: ClueT) -> float:
        """Relative importance for bundling/selection; defaults to 0."""

        _ = clue
        return 0.0

    def validator(self) -> ClueValidator:
        """Return semantic validator for this extractor."""

        return NullValidator()

    def participants(self) -> Mapping[int, list[str]]:
        """Return participants by scene; defaults to empty dict."""
        return {}


class BatchExtractor(ClueExtractor[ClueT], ABC):
    """Template for extractors that operate over scene batches via LLM jobs."""

    batch_size: int = 10
    _clue_slug: str = ""

    def __init__(self) -> None:
        super().__init__()
        self._participants: MutableMapping[int, list[str]] = {}
        self._client: Any = None
        self._batch_size: int | None = None
        self._id_counters: defaultdict[int, int] = defaultdict(int)

    @abstractmethod
    def _build_inline_requests(
        self, scenes: list[dict]
    ) -> list[types.InlinedRequestDict]:
        """Build Gemini API requests for a batch of scenes."""

    @abstractmethod
    def _parse_response(
        self, raw_payload: Any, scene_id: int
    ) -> tuple[list[str], list[ClueT]]:
        """Parse API response into (participants, clues)."""

    def participants(self) -> Mapping[int, list[str]]:
        return self._participants

    def extract(self, scene_text: str, scene_id: int) -> Sequence[ClueT]:
        return self.batch_extract([(scene_id, scene_text)])

    def batch_extract(self, items: Iterable[tuple[int, str]]) -> Sequence[ClueT]:
        scenes = [{"scene": sid, "text": txt} for sid, txt in items]
        return self._run_batch(scenes)

    def _run_batch(self, scenes: list[dict]) -> list[ClueT]:
        """Common batch processing logic using Template Method pattern."""
        outputs: list[ClueT] = []
        if not scenes:
            return outputs
        if self._client is None:
            raise ValueError(
                f"{self._clue_slug.upper()}Extractor must be configured with a client"
            )

        chunk = self._batch_size or 10
        total = (len(scenes) + chunk - 1) // chunk

        for i in range(0, len(scenes), chunk):
            sub = scenes[i : i + chunk]
            batch_idx = (i // chunk) + 1

            # Submit batch
            log_status(
                f"{self._clue_slug.upper()} batch {batch_idx}/{total}: "
                f"submitting {len(sub)} scenes"
            )
            inlined = self._build_inline_requests(sub)
            job = self._client.batches.create(
                model="gemini-2.5-flash",
                src=types.BatchJobSourceDict(inlined_requests=inlined),
                config=types.CreateBatchJobConfigDict(
                    display_name=f"{self._clue_slug}-{i // chunk:03d}",
                ),
            )

            # Poll until complete
            self._poll_job(job, batch_idx, total)

            # Process responses
            bj = self._client.batches.get(name=job.name)
            assert bj.dest is not None and bj.dest.inlined_responses is not None

            for idx, resp in enumerate(bj.dest.inlined_responses, start=1):
                result = self._process_single_response(
                    resp, sub[idx - 1], batch_idx, total, idx
                )
                if result:
                    outputs.extend(result)

        return outputs

    def _poll_job(self, job: Any, batch_idx: int, total: int) -> None:
        """Poll batch job until completion."""
        assert job.name is not None
        done_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }
        last_state: str | None = None

        while True:
            bj = self._client.batches.get(name=job.name)
            assert bj.state is not None
            state_name = bj.state.name

            if state_name != last_state:
                log_status(
                    f"{self._clue_slug.upper()} batch {batch_idx}/{total}: "
                    f"{state_name.lower()}"
                )
                last_state = state_name

            if state_name in done_states:
                if state_name != "JOB_STATE_SUCCEEDED":
                    raise RuntimeError(
                        f"{self._clue_slug.upper()} batch failed: "
                        f"{state_name} {bj.error}"
                    )
                break

            time.sleep(3)

    def _process_single_response(
        self,
        resp: Any,
        scene_dict: dict,
        batch_idx: int,
        total: int,
        idx: int,
    ) -> list[ClueT] | None:
        """Process a single response from the batch job."""
        prefix = f"{self._clue_slug.upper()} batch {batch_idx}/{total}: inline {idx}"

        if resp.error:
            log_status(f"{prefix} error -> {resp.error}")
            return None

        parsed = getattr(resp.response, "parsed", None) if resp.response else None
        raw_payload = parsed or getattr(resp.response, "text", None)
        if raw_payload is None:
            log_status(f"{prefix} empty response")
            return None

        try:
            scene_id = int(scene_dict["scene"])
            participants, clues = self._parse_response(raw_payload, scene_id)
            clues = self._assign_ids(scene_id, clues)
            if participants:
                self._participants[scene_id] = participants
            return clues
        except (ValidationError, Exception) as err:
            log_status(f"{prefix} parse error -> {err}")
            return None

    def _assign_ids(self, scene_id: int, clues: list[ClueT]) -> list[ClueT]:
        """Assign unique IDs to clues."""
        assigned: list[ClueT] = []
        for clue in clues:
            self._id_counters[scene_id] += 1
            new_id = (
                f"{self._clue_slug}_{scene_id:03d}_{self._id_counters[scene_id]:04d}"
            )
            assigned.append(clue.model_copy(update={"id": new_id}))
        return assigned


class NullValidator(ClueValidator):
    """Default validator that always passes semantic/ coherence checks."""

    def validate_semantic(self, clue: BaseClue) -> ValidationResult:  # noqa: D401
        _ = clue
        return ValidationResult.ok(level="semantic")

    def validate_coherence(
        self, clue: BaseClue, context: Mapping[str, Any] | None = None
    ) -> ValidationResult | None:  # noqa: D401
        _ = clue, context
        return None


__all__ = ["BatchExtractor", "ClueExtractor", "ClueValidator", "NullValidator"]
