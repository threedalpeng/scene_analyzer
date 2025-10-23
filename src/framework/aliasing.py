from __future__ import annotations

from collections import defaultdict

from google import genai

from clues.entity import EntityClue
from merge_entity import alias_name, alias_pair, build_alias_map, run_s2_alias
from schema import AliasGroup, AliasGroups


class AliasResolver:
    """Resolve character aliases using entity clues and LLM assistance."""

    def __init__(self, client: genai.Client | None = None) -> None:
        self._client = client

    @staticmethod
    def from_clues(entity_clues: list[EntityClue]) -> AliasGroups:
        groups: dict[str, set[str]] = defaultdict(set)
        for clue in entity_clues:
            canonical = clue.name
            groups[canonical].add(canonical)
            groups[canonical].update(clue.aliases_in_scene)
        return AliasGroups(
            groups=[
                AliasGroup(canonical=name, aliases=sorted(list(aliases)))
                for name, aliases in groups.items()
            ]
        )

    def resolve(
        self,
        unique_names: list[str],
        appearances: dict[str, list[int]],
        entity_clues: list[EntityClue] | None = None,
    ) -> tuple[AliasGroups, dict[str, str]]:
        base_groups = AliasGroups(groups=[])
        if entity_clues:
            base_groups = self.from_clues(entity_clues)

        if self._client is not None:
            llm_groups = run_s2_alias(self._client, unique_names, appearances)
        else:
            llm_groups = AliasGroups(groups=[])

        groups = AliasGroups(groups=[*base_groups.groups, *llm_groups.groups])
        return groups, build_alias_map(groups)


__all__ = [
    "AliasResolver",
    "alias_name",
    "alias_pair",
    "build_alias_map",
]
