import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import Any, Type

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, create_model

from framework.base import ClueExtractor, ClueT
from framework.pipeline import PipelineConfig
from framework.prompts import build_system_prompt
from schema import BaseClue
from utils import log_status, parse_model


class BatchExtractor(ClueExtractor[ClueT], ABC):
    """Template for extractors that operate over scene batches via LLM jobs."""

    batch_size: int = 50
    _clue_slug: str = ""

    def __init__(self) -> None:
        super().__init__()
        self._participants: MutableMapping[int, list[str]] = {}
        self._client: genai.Client | None = None
        self._batch_size: int | None = None
        self._id_counters: defaultdict[int, int] = defaultdict(int)
        self._response_schema_single: type[BaseModel] | None = None
        self._system_prompt_single: str | None = None

    @abstractmethod
    def _parse_response(
        self, raw_payload: Any, scene_id: int
    ) -> tuple[list[str], list[ClueT]]:
        """Parse API response into (participants, clues)."""

    @abstractmethod
    def get_clue_specification(self) -> dict:
        """Return configuration describing this clue extractor."""

    @abstractmethod
    def get_api_model(self) -> Type[BaseModel]:
        """Return the API response model for structured parsing."""

    def participants(self) -> Mapping[int, list[str]]:
        return self._participants

    def extract(self, scene_text: str, scene_id: int) -> Sequence[ClueT]:
        return self.batch_extract([(scene_id, scene_text)])

    def batch_extract(self, items: Iterable[tuple[int, str]]) -> Sequence[ClueT]:
        scenes = [{"scene": sid, "text": txt} for sid, txt in items]
        return self._run_batch(scenes)

    def _build_inline_requests(
        self, scenes: list[dict]
    ) -> list[types.InlinedRequestDict]:
        """Build Gemini API requests for a batch of scenes."""

        system_prompt = self._build_system_prompt_single()
        schema = self._build_response_schema()

        requests: list[types.InlinedRequestDict] = []
        for item in scenes:
            sid = int(item["scene"])
            text = str(item["text"])
            requests.append(
                types.InlinedRequestDict(
                    contents=[
                        {
                            "role": "user",
                            "parts": [{"text": self._user_prompt(sid, text)}],
                        }
                    ],
                    config=types.GenerateContentConfigDict(
                        system_instruction=system_prompt,
                        response_schema=schema,
                        response_mime_type="application/json",
                    ),
                )
            )
        return requests

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
            assert job.name
            bj = self._client.batches.get(name=job.name)
            assert bj.dest is not None and bj.dest.inlined_responses is not None

            for idx, resp in enumerate(bj.dest.inlined_responses, start=1):
                result = self._process_single_response(
                    resp, sub[idx - 1], batch_idx, total, idx
                )
                if result:
                    outputs.extend(result)

        return outputs

    def _poll_job(self, job: types.BatchJob, batch_idx: int, total: int) -> None:
        """Poll batch job until completion."""
        assert job.name
        assert self._client

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

    def _parse_clue_list(
        self, clue_list: Sequence[Mapping[str, Any] | BaseModel], scene_id: int
    ) -> tuple[list[str], list[ClueT]]:
        """Parse a subset of clues when part of a combined response."""
        serialized: list[Mapping[str, Any]] = []
        for clue in clue_list:
            if isinstance(clue, BaseModel):
                serialized.append(clue.model_dump())
            else:
                serialized.append(dict(clue))

        payload = {
            "participants": [],
            f"{self._clue_slug}_clues": serialized,
        }
        return self._parse_response(payload, scene_id)

    # ------------------------------------------------------------------
    # Prompt + schema helpers
    # ------------------------------------------------------------------
    def _build_system_prompt_single(self) -> str:
        if self._system_prompt_single is None:
            spec = self.get_clue_specification()
            self._system_prompt_single = build_system_prompt([spec])
        return self._system_prompt_single

    def _user_prompt(self, scene_id: int, text: str) -> str:
        return f"SCENE_ID: {scene_id}\nTEXT:\n{text}\n\nExtract all applicable clues."

    def _build_response_schema(self) -> Type[BaseModel]:
        if self._response_schema_single is None:
            slug = self._clue_slug
            api_model = self.get_api_model()
            fields = {
                "participants": (list[str], Field(default_factory=list)),
                f"{slug}_clues": (list[api_model], Field(default_factory=list)),
            }
            self._response_schema_single = create_model(
                f"{slug.title()}ExtractionPayload", **fields
            )
        return self._response_schema_single

    def get_prompt_section(self) -> str:
        """Legacy method for backward compatibility with prompt builders."""
        spec = self.get_clue_specification()
        sections = [
            f"## {spec['display_name']}",
            f"PURPOSE: {spec['purpose']}\n",
            "CORE CONCEPTS:",
        ]
        for name, desc in spec["concepts"]:
            sections.append(f"  - {name}: {desc}")
        special_rules = spec.get("special_rules") or []
        if special_rules:
            sections.append("\nSPECIAL RULES:")
            for rule in special_rules:
                sections.append(f"  - {rule}")
        return "\n".join(sections)


class CombinedBatchExtractor(BatchExtractor[BaseClue]):
    """Batch extractor that multiplexes several BatchExtractors into one call."""

    _clue_slug = "combined"

    def __init__(self, extractors: Sequence[BatchExtractor[Any]]) -> None:
        super().__init__()
        if not extractors:
            raise ValueError("CombinedBatchExtractor requires at least one extractor")
        self._sub_extractors: list[BatchExtractor[Any]] = list(extractors)
        self._response_schema: type[BaseModel] | None = None
        self._system_prompt: str | None = None
        self._slug_counters: defaultdict[str, defaultdict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def get_clue_specification(self) -> dict:
        """
        Combined extractor doesn't have a single specification.
        This should not be called - use sub_extractors instead.
        """
        raise NotImplementedError(
            "CombinedBatchExtractor wraps multiple extractors. "
            "Use sub_extractors to access individual specifications."
        )

    @property
    def clue_type(self) -> Type[BaseClue]:  # noqa: D401
        return BaseClue

    def registry_members(self) -> Sequence["ClueExtractor[Any]"]:
        return list(self._sub_extractors)

    @property
    def sub_extractors(self) -> Sequence[BatchExtractor[Any]]:
        return tuple(self._sub_extractors)

    def configure(self, config: "PipelineConfig") -> None:
        super().configure(config)
        if self._client is None:
            self._client = config.client
        if self._batch_size is None:
            self._batch_size = config.batch_size or 10
        if self._client is None:
            raise ValueError("CombinedBatchExtractor requires a client")
        for extractor in self._sub_extractors:
            configure = getattr(extractor, "configure", None)
            if callable(configure):
                configure(config)

    def get_prompt_section(self) -> str:  # pragma: no cover - unused for combined
        specs = [
            extractor.get_clue_specification() for extractor in self._sub_extractors
        ]
        return build_system_prompt(specs)

    def get_api_model(self) -> Type[BaseModel]:  # pragma: no cover - unused
        if self._response_schema is None:
            self._response_schema = self._build_combined_schema()
        return self._response_schema

    def _build_inline_requests(
        self, scenes: list[dict]
    ) -> list[types.InlinedRequestDict]:
        schema, system_prompt = self._ensure_assets()
        requests: list[types.InlinedRequestDict] = []
        for item in scenes:
            sid = int(item["scene"])
            text = str(item["text"])
            requests.append(
                types.InlinedRequestDict(
                    contents=[
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": f"SCENE_ID: {sid}\nTEXT:\n{text}\n\n"
                                    "Extract every requested clue type."
                                }
                            ],
                        }
                    ],
                    config=types.GenerateContentConfigDict(
                        system_instruction=system_prompt,
                        response_schema=schema,
                        response_mime_type="application/json",
                    ),
                )
            )
        return requests

    def _parse_response(
        self, raw_payload: Any, scene_id: int
    ) -> tuple[list[str], list[BaseClue]]:
        schema, _ = self._ensure_assets()
        payload_model = parse_model(schema, raw_payload)
        payload = payload_model.model_dump()
        combined_participants: set[str] = set(payload.get("participants", []) or [])
        combined_clues: list[BaseClue] = []

        for extractor in self._sub_extractors:
            slug = extractor._clue_slug
            clue_items = payload.get(f"{slug}_clues", [])
            participants, clues = extractor._parse_clue_list(clue_items, scene_id)
            combined_participants.update(participants)
            combined_clues.extend(clues)

        return sorted(combined_participants), combined_clues

    def _assign_ids(self, scene_id: int, clues: list[BaseClue]) -> list[BaseClue]:
        assigned: list[BaseClue] = []
        for clue in clues:
            slug = getattr(clue, "clue_type", "") or self._clue_slug
            counters = self._slug_counters[slug]
            counters[scene_id] += 1
            new_id = f"{slug}_{scene_id:03d}_{counters[scene_id]:04d}"
            assigned.append(clue.model_copy(update={"id": new_id}))
        return assigned

    def _build_combined_schema(self) -> type[BaseModel]:
        fields: dict[str, tuple[type, Any]] = {
            "participants": (list[str], Field(default_factory=list))
        }
        for extractor in self._sub_extractors:
            slug = extractor._clue_slug
            api_model = extractor.get_api_model()
            fields[f"{slug}_clues"] = (
                list[api_model],  # type: ignore[index]
                Field(default_factory=list),
            )
        return create_model("CombinedExtractionPayload", **fields)  # type: ignore[arg-type]

    def _build_system_prompt(self) -> str:
        specs = [
            extractor.get_clue_specification() for extractor in self._sub_extractors
        ]
        return build_system_prompt(specs)

    def _ensure_assets(self) -> tuple[type[BaseModel], str]:
        if self._response_schema is None:
            self._response_schema = self._build_combined_schema()
        if self._system_prompt is None:
            self._system_prompt = self._build_system_prompt()
        return self._response_schema, self._system_prompt
