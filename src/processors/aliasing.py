from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from google import genai

from clues.act import ActClue
from clues.entity import EntityClue
from clues.tom import ToMClue
from framework.result import PipelineResult
from merge_entity import alias_name, alias_pair, build_alias_map, run_s2_alias
from schema import AliasGroup, AliasGroups
from processors.results import AliasingResult

if TYPE_CHECKING:
    from framework.pipeline import PipelineConfig


class AliasResolver:
    """Resolve character aliases using entity clues and LLM assistance."""

    def __init__(self, client: genai.Client | None = None) -> None:
        self._client = client

    def configure(self, config: "PipelineConfig") -> None:
        if self._client is None:
            self._client = config.client

    @staticmethod
    def from_clues(result: PipelineResult) -> AliasGroups:
        groups: dict[str, set[str]] = defaultdict(set)
        for clue in result.get(EntityClue):
            canonical = clue.name
            groups[canonical].add(canonical)
            groups[canonical].update(clue.aliases_in_scene)
        return AliasGroups(
            groups=[
                AliasGroup(canonical=name, aliases=sorted(list(aliases)))
                for name, aliases in groups.items()
            ]
        )

    def __call__(self, result: PipelineResult) -> AliasingResult:
        base_groups = AliasGroups(groups=[])
        entities = result.get(EntityClue)
        if entities:
            base_groups = self.from_clues(result)

        unique_names: set[str] = set()
        appearances: dict[str, set[int]] = defaultdict(set)

        for clue in [*result.get(ActClue), *result.get(ToMClue)]:
            pair = clue.pair
            unique_names.update(pair)
            appearances[pair[0]].add(clue.scene)
            appearances[pair[1]].add(clue.scene)

        unique_list = sorted(unique_names)
        appearances_int = {k: sorted(v) for k, v in appearances.items()}

        llm_groups = AliasGroups(groups=[])
        if self._client is not None and unique_list:
            llm_groups = run_s2_alias(self._client, unique_list, appearances_int)

        groups = AliasGroups(groups=[*base_groups.groups, *llm_groups.groups])
        alias_map = build_alias_map(groups)
        return AliasingResult(alias_groups=groups, alias_map=alias_map)


__all__ = ["AliasResolver", "alias_name", "alias_pair", "build_alias_map"]
