import argparse
import json
from pathlib import Path

from client import make_client
from clues.act import ActClue, ActExtractor
from clues.entity import EntityClue, EntityExtractor
from clues.temporal import TemporalClue, TemporalExtractor
from clues.tom import ToMClue, ToMExtractor
from framework import Pipeline, PipelineConfig
from processors.aliasing import AliasResolver
from processors.results import AliasingResult, SynthesisResult, TemporalResult
from processors.synthesis import DyadSynthesizer
from processors.temporal import TemporalReconstructor
from schema import LLMAdjudication
from utils import ensure_dir, jsonl_write, log_status


def _load_scenes(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_metadata(path: Path | None) -> dict[int, dict]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def _serialize_validation(results):
    serialized = []
    for clue, records in results:
        serialized.append(
            {
                "clue": clue.model_dump(),
                "results": [r.model_dump() for r in records],
            }
        )
    return serialized


def _serialize_dyads(dyads: dict[tuple[str, str], LLMAdjudication]) -> list[dict]:
    out: list[dict] = []
    for pair, adjudication in dyads.items():
        out.append({"pair": list(pair), "adjudication": adjudication.model_dump()})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run narrative analysis pipeline")
    parser.add_argument("scenes", type=Path, help="Path to scenes JSONL input")
    parser.add_argument("out", type=Path, help="Output directory")
    parser.add_argument(
        "--metadata", type=Path, default=None, help="Optional fabula metadata JSON"
    )
    parser.add_argument(
        "--batch", type=int, default=10, help="Batch size for Gemini requests"
    )
    args = parser.parse_args()

    ensure_dir(args.out)
    client = make_client()

    config = PipelineConfig(client=client, batch_size=args.batch)
    pipeline = (
        Pipeline(config)
        .extract(ActExtractor)
        .extract(ToMExtractor)
        .extract(TemporalExtractor)
        .extract(EntityExtractor)
        .process(AliasResolver())
        .process(TemporalReconstructor())
        .process(DyadSynthesizer())
    )

    scenes = _load_scenes(args.scenes)
    metadata = _load_metadata(args.metadata)

    log_status("Starting pipeline run")
    pipeline_result = pipeline.run(scenes, metadata=metadata)

    jsonl_write(
        args.out / "act_clues.jsonl",
        [clue.model_dump() for clue in pipeline_result.get(ActClue)],
    )
    jsonl_write(
        args.out / "tom_clues.jsonl",
        [clue.model_dump() for clue in pipeline_result.get(ToMClue)],
    )
    jsonl_write(
        args.out / "temporal_clues.jsonl",
        [clue.model_dump() for clue in pipeline_result.get(TemporalClue)],
    )
    jsonl_write(
        args.out / "entity_clues.jsonl",
        [clue.model_dump() for clue in pipeline_result.get(EntityClue)],
    )
    jsonl_write(
        args.out / "validation.jsonl",
        _serialize_validation(pipeline_result.validation),
    )

    aliasing = pipeline_result.get(AliasingResult)
    if aliasing:
        (args.out / "alias_groups.json").write_text(
            aliasing.alias_groups.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (args.out / "alias_map.json").write_text(
            json.dumps(aliasing.alias_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    temporal = pipeline_result.get(TemporalResult)
    if temporal:
        (args.out / "fabula_rank.json").write_text(
            json.dumps(temporal.fabula_rank, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    synthesis = pipeline_result.get(SynthesisResult)
    if synthesis:
        jsonl_write(
            args.out / "acts_representative.jsonl",
            [clue.model_dump() for clue in synthesis.acts_representative],
        )
        jsonl_write(
            args.out / "acts_directed.jsonl",
            [clue.model_dump() for clue in synthesis.acts_directed],
        )
        jsonl_write(
            args.out / "dyad_results.jsonl",
            _serialize_dyads(synthesis.dyad_results),
        )

    log_status("Pipeline completed")


if __name__ == "__main__":
    main()
