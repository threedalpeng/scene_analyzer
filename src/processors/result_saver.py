from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Type

from clues.act import ActClue
from clues.entity import EntityClue
from clues.temporal import TemporalClue
from clues.tom import ToMClue
from framework.result import PipelineResult
from processors.results import AliasingResult, SynthesisResult, TemporalResult
from schema import BaseClue, LLMAdjudication, ValidationResult
from utils import ensure_dir, jsonl_write, log_status


class ResultSaver:
    """Persist pipeline outputs (clues, validations, processors) to disk."""

    def __init__(self, output_dir: Path, *, ensure_directory: bool = True) -> None:
        self.output_dir = output_dir
        self.ensure_directory = ensure_directory

    def save(self, result: PipelineResult) -> None:
        """Write all known artifacts derived from `result`."""
        if self.ensure_directory:
            ensure_dir(self.output_dir)

        log_status(f"Saving pipeline artifacts to {self.output_dir}")
        self._write_clue_files(result)
        self._write_validation(result)
        self._write_aliasing(result.get(AliasingResult))
        self._write_temporal(result.get(TemporalResult))
        self._write_synthesis(result.get(SynthesisResult))

    def __call__(self, result: PipelineResult) -> None:
        """Allow ResultSaver to be used as a pipeline processor."""
        self.save(result)

    # --- clue + validation writers -------------------------------------------------
    def _write_clue_files(self, result: PipelineResult) -> None:
        mapping: Mapping[Type[BaseClue], str] = {
            ActClue: "act_clues.jsonl",
            ToMClue: "tom_clues.jsonl",
            TemporalClue: "temporal_clues.jsonl",
            EntityClue: "entity_clues.jsonl",
        }
        for clue_type, filename in mapping.items():
            records = [clue.model_dump() for clue in result.get(clue_type)]
            jsonl_write(self.output_dir / filename, records)

    def _write_validation(self, result: PipelineResult) -> None:
        payload = _serialize_validation(result.validation)
        jsonl_write(self.output_dir / "validation.jsonl", payload)

    # --- processor outputs ---------------------------------------------------------
    def _write_aliasing(self, aliasing: AliasingResult | None) -> None:
        if aliasing is None:
            return
        (self.output_dir / "alias_groups.json").write_text(
            aliasing.alias_groups.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (self.output_dir / "alias_map.json").write_text(
            json.dumps(aliasing.alias_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_temporal(self, temporal: TemporalResult | None) -> None:
        if temporal is None:
            return
        (self.output_dir / "fabula_rank.json").write_text(
            json.dumps(temporal.fabula_rank, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_synthesis(self, synthesis: SynthesisResult | None) -> None:
        if synthesis is None:
            return
        jsonl_write(
            self.output_dir / "acts_representative.jsonl",
            [clue.model_dump() for clue in synthesis.acts_representative],
        )
        jsonl_write(
            self.output_dir / "acts_directed.jsonl",
            [clue.model_dump() for clue in synthesis.acts_directed],
        )
        jsonl_write(
            self.output_dir / "dyad_results.jsonl",
            _serialize_dyads(synthesis.dyad_results),
        )


# ---------------------------------------------------------------------------
# Helper serialization routines reused between CLI + processors
# ---------------------------------------------------------------------------
def _serialize_validation(
    records: Iterable[tuple[BaseClue, Iterable[ValidationResult]]]
) -> list[dict]:
    serialized: list[dict] = []
    for clue, validations in records:
        serialized.append(
            {
                "clue": clue.model_dump(),
                "results": [validation.model_dump() for validation in validations],
            }
        )
    return serialized


def _serialize_dyads(dyads: Mapping[tuple[str, str], LLMAdjudication]) -> list[dict]:
    out: list[dict] = []
    for pair, adjudication in dyads.items():
        out.append({"pair": list(pair), "adjudication": adjudication.model_dump()})
    return out


__all__ = ["ResultSaver"]
