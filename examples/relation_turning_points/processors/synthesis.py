from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Type

from clues.act import ActClue, act_score, bundle_same_segment, explode_directed
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
You adjudicate relationship state across segments for a dyad (character pair).
Use ONLY the provided clue packs and return strict JSON per schema.
You can update the relation states using evidences if they don't match.

CORE CONCEPTS:

Turning: A qualitative change in relationship state.
Category (required): Reversal, Strengthening, Commitment, Dissolution.
Pattern: Free-form label describing the domain-specific pattern (e.g., "betrayal").

Turning Detection Logic:
  1. Identify candidate when there is a category change meeting the threshold.
  2. Threshold: stakes="major" OR (salience="high" AND durability="persistent") OR referenced_segments present.
  3. Verify representative act exists.
  4. Check for implicit turns: IntendsTo/DesiresFor followed within ≤2 segments by consistent action.

Pre/Post State:
  - pre_state: valence/durability before the turn.
  - post_state: valence/durability after the representative act.
  - Include optional label to clarify the narrative state if helpful.

Final Relation:
  - Summarize the overall relationship (valence + optional label).

ID Rules:
  - Preserve ids exactly as provided.
  - representative_act_id must match an act id from input.
  - caused_by_tom_ids must reference ToM ids when used.

Output JSON must match the provided schema exactly.
"""


def _dyad_user_payload(
    dossier_lines: list[str], tom_lines: list[str], causal_json: dict, stats_json: dict
) -> str:
    return (
        "DYAD_DOSSIER:\n"
        + "\n".join(dossier_lines)
        + "\n\nTOM_THREADS:\n"
        + "\n".join(tom_lines)
        + "\n\nCAUSAL_CHAIN:\n"
        + json.dumps(causal_json, ensure_ascii=False, indent=2)
        + "\n\nSTATS:\n"
        + json.dumps(stats_json, ensure_ascii=False, indent=2)
    ).strip()


@dataclass
class DyadBag:
    pair: tuple[str, str]
    acts_rep: list[ActClue] = field(default_factory=list)
    acts_all: list[ActClue] = field(default_factory=list)
    toms: list[ToMClue] = field(default_factory=list)


class DyadAnalysis(BaseModel):
    char1: str
    char2: str
    adjudication: LLMAdjudication


class SynthesisResult(BaseModel):
    acts_representative: list[ActClue]
    acts_directed: list[ActClue]
    dyads: list[DyadAnalysis]


def build_bags(
    toms: list[ToMClue], acts_rep: list[ActClue], acts_all: list[ActClue]
) -> dict[tuple[str, str], DyadBag]:
    bags: dict[tuple[str, str], DyadBag] = {}

    def ensure(pair: tuple[str, str]) -> DyadBag:
        if pair not in bags:
            bags[pair] = DyadBag(pair=pair)
        return bags[pair]

    for act in acts_rep:
        ensure(act.pair).acts_rep.append(act)
    for act in acts_all:
        ensure(act.pair).acts_all.append(act)
    for tom in toms:
        ensure(tom.pair).toms.append(tom)

    for bag in bags.values():
        bag.acts_rep.sort(key=lambda x: x.segment)
        bag.acts_all.sort(key=lambda x: x.segment)
        bag.toms.sort(key=lambda x: x.segment)

    return bags


def _sample_dossier(acts_rep: list[ActClue]) -> list[ActClue]:
    if not acts_rep:
        return []
    items = sorted(acts_rep, key=act_score, reverse=True)
    rep = items[0]
    supports = [a for a in items[1:] if a.valence != rep.valence][:1]
    if len(supports) < 1 and len(items) > 1:
        supports = [items[1]]
    return [rep, *supports]


def _packs_for_pair(bag: DyadBag) -> tuple[list[str], list[str], dict, dict]:
    dossier = _sample_dossier(bag.acts_rep)
    dossier_lines = [json.dumps(a.model_dump(), ensure_ascii=False) for a in dossier]
    tom_lines = [json.dumps(t.model_dump(), ensure_ascii=False) for t in bag.toms[-6:]]

    causal_edges = []
    for act in bag.acts_all:
        for ref in act.referenced_segments:
            causal_edges.append(
                {
                    "src_segment": act.segment,
                    "dst_segment": int(ref),
                    "via": [act.id],
                }
            )
    causal_json = {"edges": causal_edges[:50]}

    from collections import Counter

    valence = Counter(act.valence for act in bag.acts_all)
    sal = Counter(act.axes.salience for act in bag.acts_all)
    stakes = Counter(act.axes.stakes for act in bag.acts_all)
    durability = Counter(act.axes.durability for act in bag.acts_all)
    coerced = sum(1 for act in bag.acts_all if act.axes.volition == "coerced")

    stats = {
        "acts_count": len(bag.acts_all),
        "valence_freq": dict(valence),
        "salience_freq": dict(sal),
        "stakes_freq": dict(stakes),
        "durability_freq": dict(durability),
        "coerced_ratio": (coerced / len(bag.acts_all)) if bag.acts_all else 0.0,
    }

    return dossier_lines, tom_lines, causal_json, stats


class DyadSynthesizer(Processor):
    """
    Synthesize relationship timelines for character dyads using Gemini batch analysis.

    Requirements:
        - client: Required (configure() raises ValueError if unavailable).
        - batch_size: Optional override for batch submission size (default 10).

    Input: ActClue, ToMClue
    Output: SynthesisResult with representative acts, directed acts, and adjudicated dyads.
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

        acts_representative = bundle_same_segment(acts)
        acts_directed = explode_directed(acts)
        bags = build_bags(toms, acts_representative, acts_directed)
        adjudication = self._run_batch(bags.items())
        dyads = []
        for pair, payload in adjudication.items():
            sorted_pair = tuple(sorted(pair))
            dyads.append(
                DyadAnalysis(
                    char1=sorted_pair[0], char2=sorted_pair[1], adjudication=payload
                )
            )
        return SynthesisResult(
            acts_representative=acts_representative,
            acts_directed=acts_directed,
            dyads=dyads,
        )

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
            dossier, toms, causal, stats = _packs_for_pair(bag)
            requests.append(
                types.InlinedRequestDict(
                    contents=[
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": _dyad_user_payload(
                                        dossier, toms, causal, stats
                                    )
                                }
                            ],
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


__all__ = [
    "DyadSynthesizer",
    "DyadBag",
    "DyadAnalysis",
    "SynthesisResult",
    "build_bags",
]
