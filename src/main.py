import argparse
import json
from pathlib import Path

from client import make_client
from clues.act import ActExtractor
from clues.entity import EntityExtractor
from clues.temporal import TemporalExtractor
from clues.tom import ToMExtractor
from framework import Pipeline, PipelineConfig
from processors.aliasing import AliasResolver
from processors.result_saver import ResultSaver
from processors.synthesis import DyadSynthesizer
from processors.temporal import TemporalReconstructor
from utils import ensure_dir, log_status


def _load_scenes(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_metadata(path: Path | None) -> dict[int, dict]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


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
        .extract([ActExtractor, ToMExtractor, TemporalExtractor, EntityExtractor])
        .process(AliasResolver())
        .process(TemporalReconstructor())
        .process(DyadSynthesizer())
        .process(ResultSaver(args.out))
    )

    scenes = _load_scenes(args.scenes)
    metadata = _load_metadata(args.metadata)

    log_status("Starting pipeline run")
    pipeline.run(scenes, metadata=metadata)

    log_status("Pipeline completed")


if __name__ == "__main__":
    main()
