from __future__ import annotations

import hashlib
import json
from typing import Any, Type


def type_name(cls: Type[Any]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def compute_digest(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


__all__ = [
    "compute_digest",
    "type_name",
]
