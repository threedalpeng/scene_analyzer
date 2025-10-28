import json
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Type, TypeVar

from google import genai
from pydantic import BaseModel


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def log_status(message: str) -> None:
    """
    Print a timestamped status line so long pipeline runs can be monitored in the shell.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def jsonl_write(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def norm_pair(a: str, b: str) -> tuple[str, str]:
    a, b = sorted((a, b))
    return a, b


T = TypeVar("T", bound=BaseModel)


def parse_model(model: Type[T], payload: str | dict[str, Any] | BaseModel) -> T:
    """
    Convert raw JSON text or dict (including already-parsed SDK payloads) into model instances.
    """
    if isinstance(payload, str):
        data = json.loads(payload)
    elif isinstance(payload, BaseModel):
        data = payload.model_dump()
    else:
        data = payload
    return model.model_validate(data)


def make_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your environment."
        )
    client = genai.Client(api_key=api_key)
    return client
