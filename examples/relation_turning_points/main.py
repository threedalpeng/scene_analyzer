import argparse
import json
from pathlib import Path
from typing import Any

from clues.act import ActExtractor
from clues.tom import ToMExtractor
from processors.aliasing import AliasResolver
from processors.result_saver import ResultSaver
from processors.synthesis import DyadSynthesizer

from framework.core import Pipeline, PipelineConfig
from framework.utils import ensure_dir, log_status, make_client


def _load_segments(path: Path) -> list[dict[str, Any]]:
    def _coerce_segment_id(raw: Any, fallback: int) -> int:
        if raw is None:
            return fallback
        try:
            return int(raw)
        except (TypeError, ValueError):
            return fallback

    def _coerce_text(entry: dict[str, Any]) -> str:
        if "text" in entry and isinstance(entry["text"], str):
            return entry["text"]
        for key in ("content", "body", "segment_text"):
            value = entry.get(key)
            if isinstance(value, str):
                entry["text"] = value
                return value
        raise KeyError("segment entry is missing a text field")

    normalized: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            entry = dict(json.loads(stripped))

            segment_id = entry.get(
                "segment",
                entry.get(
                    "segment_id",
                    entry.get("segment_index", entry.get("id")),
                ),
            )
            entry["segment"] = _coerce_segment_id(segment_id, idx)
            entry["text"] = _coerce_text(entry)
            normalized.append(entry)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Run narrative analysis pipeline")
    parser.add_argument("segments", type=Path, help="Path to segments JSONL input")
    parser.add_argument("out", type=Path, help="Output directory")
    parser.add_argument(
        "--batch", type=int, default=50, help="Batch size for Gemini requests"
    )
    args = parser.parse_args()

    ensure_dir(args.out)
    client = make_client()

    config = PipelineConfig(client=client, batch_size=args.batch)
    pipeline = (
        Pipeline(config)
        .extract([ActExtractor, ToMExtractor])
        .process(AliasResolver())
        .process(DyadSynthesizer())
        .process(ResultSaver(args.out))
    )

    segments = _load_segments(args.segments)

    log_status("Starting pipeline run")
    pipeline.run(segments)

    log_status("Pipeline completed")


if __name__ == "__main__":
    main()
