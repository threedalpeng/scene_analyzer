import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from schema import ActClue


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


STAKE_RANK = {"major": 3, "moderate": 2, "minor": 1}
SALIENCE_RANK = {"high": 3, "medium": 2, "low": 1}
DUR_RANK = {"persistent": 3, "temporary": 2, "momentary": 1}


def act_score(a: ActClue) -> int:
    return (
        100 * STAKE_RANK.get(a.axes.stakes, 0)
        + 10 * SALIENCE_RANK.get(a.axes.salience, 0)
        + DUR_RANK.get(a.axes.durability, 0)
    )


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
