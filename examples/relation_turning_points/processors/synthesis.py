from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Type

from clues.act import ActClue, act_score
from clues.tom import ToMClue
from google import genai
from google.genai import types
from pydantic import BaseModel
from schema import LLMAdjudication

from framework.core.processor import Processor
from framework.core.result import PipelineResult
from framework.utils import log_status, parse_model

if TYPE_CHECKING:
    from framework.core.pipeline import PipelineConfig


DYAD_SYSTEM_PROMPT = """
You analyze relationship change between two characters.

INPUT JSON PROVIDES:
  - acts: Chronological action clues (max 30) between the pair.
  - toms: Theory-of-mind clues (max 20) such as beliefs, feelings, intentions.
  - statistics: Aggregate counts for the full data (not just the sample).

TASK:
  1. Examine the earliest ToMs and acts (first 3-5 segments) to infer the initial relationship state.
  2. Identify turning points where the relationship state (valence and/or durability) shifts durably.
  3. For the first turning point, use your inferred initial state as pre_state.

TURNING POINT REQUIREMENTS:
  1. Clear before/after difference in relationship state.
  2. Representative act drawn from the provided `acts` (exact ID match required).
  3. Evidence grounded in the supplied clues (avoid speculation).
  4. Pre/post states derived from evidence within ±5 segments of the turning event.

CATEGORIES:
  - Reversal: Valence flips (positive ↔ negative).
  - Strengthening: Same valence, durability increases.
  - Dissolution: Relationship breaks down or hostility becomes irreversible.
  - Commitment: Relationship solidifies or deepens positively.

OUTPUT RULES:
  - Derive initial state from textual evidence; if insufficient, omit the turning point.
  - Representative act IDs must exactly match one of the supplied `acts`.
  - event_roles must label relevant acts/ToMs as "turning" or "supporting".
  - caused_by_tom_ids must list ToM IDs that causally motivate the representative act.
  - If evidence is insufficient, omit the turning point rather than guessing.

Return valid JSON conforming to LLMAdjudication schema.
""".strip()


class DyadSynthesizer(Processor):
    """
    Synthesize relationship timelines for character dyads using Gemini batch analysis.

    Requirements:
        - client: Required (configure() raises ValueError if unavailable).
        - batch_size: Optional override for batch submission size (default 10).

    Input: ActClue, ToMClue
    Output: SynthesisResult containing adjudicated dyads.
    """

    def __init__(
        self, client: genai.Client | None = None, *, batch_size: int | None = None
    ) -> None:
        self._client = client
        self._batch_size = batch_size or 10

    def configure(self, config: "PipelineConfig") -> None:
        if self._client is None:
            self._client = config.client
        if config.batch_size is not None:
            self._batch_size = config.batch_size
        if self._client is None:
            raise ValueError(
                "DyadSynthesizer requires a client; none provided in config"
            )

    def __call__(self, result: PipelineResult) -> SynthesisResult:
        acts = result.get_clues(ActClue)
        toms = result.get_clues(ToMClue)

        bags = build_bags(toms, acts)
        adjudication = self._run_batch(bags.items())
        dyads = []
        for pair, payload in adjudication.items():
            sorted_pair = tuple(sorted(pair))
            dyads.append(
                DyadAnalysis(
                    char1=sorted_pair[0], char2=sorted_pair[1], adjudication=payload
                )
            )
        return SynthesisResult(dyads=dyads)

    @property
    def result_type(self) -> Type[SynthesisResult]:
        return SynthesisResult

    def _run_batch(
        self, items: Iterable[tuple[tuple[str, str], DyadBag]]
    ) -> dict[tuple[str, str], LLMAdjudication]:
        items_list = list(items)
        if not items_list:
            return {}

        if self._client is None:
            raise ValueError(
                "DyadSynthesizer must be configured with a client before use"
            )

        results: dict[tuple[str, str], LLMAdjudication] = {}
        chunk = self._batch_size

        for i in range(0, len(items_list), chunk):
            subset = items_list[i : i + chunk]
            batch_idx = (i // chunk) + 1
            log_status(
                f"SYN batch {batch_idx}: submitting {len(subset)} dyads to Gemini"
            )
            requests, order = self._build_inline_requests(subset)
            job = self._client.batches.create(
                model="gemini-2.5-flash",
                src=types.BatchJobSourceDict(inlined_requests=requests),
                config=types.CreateBatchJobConfigDict(
                    display_name=f"syn-{i // chunk:03d}",
                ),
            )

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
                    log_status(f"SYN batch {batch_idx}: {state_name.lower()}")
                    last_state = state_name
                if state_name in done_states:
                    if state_name != "JOB_STATE_SUCCEEDED":
                        raise RuntimeError(f"SYN batch failed: {state_name} {bj.error}")
                    break
                time.sleep(3)

            assert bj.dest is not None and bj.dest.inlined_responses is not None
            for idx, resp in enumerate(bj.dest.inlined_responses, start=1):
                pair = order[idx - 1]
                if resp.error:
                    log_status(
                        f"SYN batch {batch_idx}: inline {idx} error -> {resp.error}"
                    )
                    continue
                parsed = (
                    getattr(resp.response, "parsed", None) if resp.response else None
                )
                raw_payload = parsed or getattr(resp.response, "text", None)
                if raw_payload is None:
                    log_status(f"SYN batch {batch_idx}: inline {idx} empty response")
                    continue

                adjudication = parse_model(LLMAdjudication, raw_payload)
                results[pair] = adjudication

        return results

    def _build_inline_requests(
        self, items: list[tuple[tuple[str, str], DyadBag]]
    ) -> tuple[list[types.InlinedRequestDict], list[tuple[str, str]]]:
        requests: list[types.InlinedRequestDict] = []
        order: list[tuple[str, str]] = []
        for pair, bag in items:
            payload_json = _build_payload(bag)
            requests.append(
                types.InlinedRequestDict(
                    contents=[
                        {
                            "role": "user",
                            "parts": [{"text": payload_json}],
                        }
                    ],
                    config=types.GenerateContentConfigDict(
                        system_instruction=DYAD_SYSTEM_PROMPT,
                        response_schema=LLMAdjudication,
                        response_mime_type="application/json",
                    ),
                )
            )
            order.append(pair)
        return requests, order


@dataclass
class DyadBag:
    pair: tuple[str, str]
    acts: list[ActClue] = field(default_factory=list)
    toms: list[ToMClue] = field(default_factory=list)


class DyadAnalysis(BaseModel):
    char1: str
    char2: str
    adjudication: LLMAdjudication


class SynthesisResult(BaseModel):
    dyads: list[DyadAnalysis]


def build_bags(
    toms: list[ToMClue], acts: list[ActClue]
) -> dict[tuple[str, str], DyadBag]:
    bags: dict[tuple[str, str], DyadBag] = {}

    def ensure(pair: tuple[str, str]) -> DyadBag:
        if pair not in bags:
            bags[pair] = DyadBag(pair=pair)
        return bags[pair]

    for act in acts:
        ensure(act.pair).acts.append(act)
    for tom in toms:
        ensure(tom.pair).toms.append(tom)

    for bag in bags.values():
        bag.acts.sort(key=lambda x: x.segment)
        bag.toms.sort(key=lambda x: x.segment)

    return bags


def _select_acts_for_dossier(
    acts: list[ActClue],
    *,
    limit: int = 30,
) -> list[ActClue]:
    if not acts:
        return []

    ordered = sorted(acts, key=lambda a: a.segment)
    selected: dict[str, ActClue] = {}

    def _add(group: Iterable[ActClue]) -> None:
        for act in group:
            if act.id not in selected:
                selected[act.id] = act

    majors = [act for act in ordered if act.axes.stakes == "major"]
    _add(sorted(majors, key=act_score, reverse=True))

    unique_segments = sorted(set(act.segment for act in ordered))
    if unique_segments:
        early_segs = set(unique_segments[:2])
        late_segs = set(unique_segments[-2:])
        boundary_acts = [
            act for act in ordered if act.segment in early_segs.union(late_segs)
        ]
        _add(boundary_acts)

    if len(selected) < limit:
        remaining = [act for act in ordered if act.id not in selected]
        remaining.sort(key=act_score, reverse=True)
        for act in remaining:
            selected[act.id] = act
            if len(selected) >= limit:
                break

    return sorted(selected.values(), key=lambda a: a.segment)


def _select_toms_for_dossier(
    toms: list[ToMClue],
    selected_acts: list[ActClue],
    *,
    limit: int = 20,
    context_window: int = 1,
) -> list[ToMClue]:
    if not toms:
        return []

    ordered = sorted(toms, key=lambda t: t.segment)
    selected: dict[str, ToMClue] = {}

    def _add(group: Iterable[ToMClue]) -> None:
        for tom in group:
            if tom.id not in selected:
                selected[tom.id] = tom

    all_segments = sorted({t.segment for t in ordered})

    if all_segments:
        early_segs = set(all_segments[:3])
        early_toms = [t for t in ordered if t.segment in early_segs]
        _add(early_toms)

    intends = [tom for tom in ordered if tom.kind == "IntendsTo"]
    _add(intends)

    if selected_acts:
        act_segments = {act.segment for act in selected_acts}
        relevant_segments: set[int] = set()
        for seg in act_segments:
            for offset in range(-context_window, context_window + 1):
                relevant_segments.add(seg + offset)

        priority = {
            "IntendsTo": 4,
            "DesiresFor": 3,
            "FeelsTowards": 2,
            "BelievesAbout": 1,
        }
        relevant_toms = [t for t in ordered if t.segment in relevant_segments]
        relevant_toms.sort(key=lambda t: (-priority.get(t.kind, 0), t.segment))
        _add(relevant_toms)

    if len(selected) < limit and all_segments:
        late_segs = set(all_segments[-2:])
        late_toms = [t for t in ordered if t.segment in late_segs]
        _add(late_toms)

    if len(selected) < limit:
        priority = {
            "IntendsTo": 4,
            "DesiresFor": 3,
            "FeelsTowards": 2,
            "BelievesAbout": 1,
        }
        remaining = [tom for tom in ordered if tom.id not in selected]
        remaining.sort(key=lambda t: (-priority.get(t.kind, 0), t.segment))
        for tom in remaining:
            selected[tom.id] = tom
            if len(selected) >= limit:
                break

    return sorted(selected.values(), key=lambda t: t.segment)


def _compute_stats(acts: list[ActClue]) -> dict[str, int | float]:
    from collections import Counter

    valence = Counter(act.valence for act in acts)
    stakes = Counter(act.axes.stakes for act in acts)

    return {
        "total_acts": len(acts),
        "valence_positive": valence.get("positive", 0),
        "valence_negative": valence.get("negative", 0),
        "stakes_major": stakes.get("major", 0),
        "stakes_moderate": stakes.get("moderate", 0),
        "stakes_minor": stakes.get("minor", 0),
    }


def _build_payload(bag: DyadBag) -> str:
    selected_acts = _select_acts_for_dossier(bag.acts)
    selected_toms = _select_toms_for_dossier(bag.toms, selected_acts)
    stats = _compute_stats(bag.acts)

    payload = {
        "pair": {"characters": [bag.pair[0], bag.pair[1]]},
        "acts": [act.model_dump() for act in selected_acts],
        "toms": [tom.model_dump() for tom in selected_toms],
        "statistics": stats,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = [
    "DyadSynthesizer",
    "DyadBag",
    "DyadAnalysis",
    "SynthesisResult",
    "build_bags",
]
