import argparse
import json
from pathlib import Path

from client import make_client
from clues.act import ActExtractor, bundle_same_scene, explode_directed
from clues.entity import EntityExtractor
from clues.registry import ClueRegistry
from clues.temporal import TemporalExtractor
from clues.tom import ToMExtractor
from framework.aliasing import AliasResolver
from framework.orchestrator import Pipeline
from framework.synthesis import DyadSynthesizer
from framework.temporal import FabulaReconstructor
from framework.validation import ValidationPipeline
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


def _serialize_dyads(dyads: dict[tuple[str, str], object]) -> list[dict]:
    out: list[dict] = []
    for pair, adjudication in dyads.items():
        model = (
            adjudication.model_dump()
            if hasattr(adjudication, "model_dump")
            else adjudication
        )
        out.append({"pair": list(pair), "adjudication": model})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run narrative analysis pipeline")
    parser.add_argument("scenes", type=Path, help="Path to scenes JSONL input")
    parser.add_argument("out", type=Path, help="Output directory")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional fabula metadata JSON")
    parser.add_argument("--batch", type=int, default=10, help="Batch size for Gemini requests")
    args = parser.parse_args()

    ensure_dir(args.out)
    client = make_client()

    registry = ClueRegistry()
    act_extractor = ActExtractor(client, batch_size=args.batch)
    tom_extractor = ToMExtractor(client, batch_size=args.batch)
    temporal_extractor = TemporalExtractor(client, batch_size=args.batch)
    entity_extractor = EntityExtractor(client, batch_size=args.batch)
    registry.register_many([
        act_extractor,
        tom_extractor,
        temporal_extractor,
        entity_extractor,
    ])

    validator = ValidationPipeline(registry)
    fabula = FabulaReconstructor()
    alias_resolver = AliasResolver(client)
    synthesizer = DyadSynthesizer(client, batch_size=args.batch)

    pipeline = Pipeline(
        registry=registry,
        validator=validator,
        temporal=fabula,
        alias_resolver=alias_resolver,
        synthesizer=synthesizer,
        act_bundle=bundle_same_scene,
        act_explode=explode_directed,
    )

    scenes = _load_scenes(args.scenes)
    metadata = _load_metadata(args.metadata)

    log_status("Starting pipeline run")
    result = pipeline.run(scenes, metadata=metadata)

    jsonl_write(
        args.out / "act_clues.jsonl",
        [clue.model_dump() for clue in result["act_clues"]],
    )
    jsonl_write(
        args.out / "tom_clues.jsonl",
        [clue.model_dump() for clue in result["tom_clues"]],
    )
    jsonl_write(
        args.out / "temporal_clues.jsonl",
        [clue.model_dump() for clue in result["temporal_clues"]],
    )
    jsonl_write(
        args.out / "entity_clues.jsonl",
        [clue.model_dump() for clue in result["entity_clues"]],
    )
    jsonl_write(
        args.out / "validation.jsonl",
        _serialize_validation(result["validation"]),
    )
    (args.out / "fabula_rank.json").write_text(
        json.dumps(result["fabula_rank"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out / "alias_groups.json").write_text(
        result["alias_groups"].model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out / "alias_map.json").write_text(
        json.dumps(result["alias_map"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    jsonl_write(
        args.out / "acts_representative.jsonl",
        [clue.model_dump() for clue in result["acts_representative"]],
    )
    jsonl_write(
        args.out / "acts_directed.jsonl",
        [clue.model_dump() for clue in result["acts_directed"]],
    )
    jsonl_write(
        args.out / "dyad_results.jsonl",
        _serialize_dyads(result["dyad_results"]),
    )

    log_status("Pipeline completed")


if __name__ == "__main__":
    main()
