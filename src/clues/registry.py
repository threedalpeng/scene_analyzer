from __future__ import annotations

from typing import Dict, Iterable, Iterator

from clues.base import ClueExtractor


class ClueRegistry:
    """Simple registry that keeps track of available extractors by id."""

    def __init__(self) -> None:
        self._extractors: Dict[str, ClueExtractor] = {}

    def register(self, extractor: ClueExtractor) -> None:
        cid = extractor.clue_type
        if cid in self._extractors:
            raise ValueError(f"clue '{cid}' already registered")
        self._extractors[cid] = extractor

    def register_many(self, extractors: Iterable[ClueExtractor]) -> None:
        for ex in extractors:
            self.register(ex)

    def get(self, clue_type: str) -> ClueExtractor:
        try:
            return self._extractors[clue_type]
        except KeyError as err:
            raise KeyError(f"no extractor registered for '{clue_type}'") from err

    def items(self) -> Iterator[tuple[str, ClueExtractor]]:
        return iter(self._extractors.items())

    def values(self) -> Iterator[ClueExtractor]:
        return iter(self._extractors.values())

    def __contains__(self, clue_type: str) -> bool:
        return clue_type in self._extractors

    def __len__(self) -> int:
        return len(self._extractors)
