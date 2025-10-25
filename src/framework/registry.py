from __future__ import annotations

from typing import Dict, Iterable, Iterator, Type

from framework.base import ClueExtractor
from schema import BaseClue


class ClueRegistry:
    """Registry that keeps track of available clue extractors."""

    def __init__(self) -> None:
        self._extractors: Dict[type[BaseClue], ClueExtractor] = {}

    def register(self, extractor: ClueExtractor) -> None:
        clue_cls = extractor.clue_type
        if not isinstance(clue_cls, type) or not issubclass(clue_cls, BaseClue):
            raise TypeError("clue_type must be a BaseClue subclass")
        if clue_cls in self._extractors:
            raise ValueError(f"clue '{clue_cls.__name__}' already registered")
        self._extractors[clue_cls] = extractor

    def register_many(self, extractors: Iterable[ClueExtractor]) -> None:
        for ex in extractors:
            self.register(ex)

    def get(self, clue_type: Type[BaseClue]) -> ClueExtractor:
        try:
            return self._extractors[clue_type]
        except KeyError as err:
            raise KeyError(
                f"no extractor registered for '{clue_type.__name__}'"
            ) from err

    def items(self) -> Iterator[tuple[type[BaseClue], ClueExtractor]]:
        return iter(self._extractors.items())

    def values(self) -> Iterator[ClueExtractor]:
        return iter(self._extractors.values())

    def __contains__(self, clue_type: Type[BaseClue]) -> bool:
        return clue_type in self._extractors

    def __len__(self) -> int:
        return len(self._extractors)
