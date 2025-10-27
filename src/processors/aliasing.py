from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from clues.act import ActClue
from clues.entity import EntityClue
from clues.tom import ToMClue
from framework.result import PipelineResult
from processors.types import AliasingResult
from schema import AliasGroup, AliasGroups
from utils import norm_pair, parse_model

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
        base_groups = self._extract_from_entity_clues(result)

        llm_groups = AliasGroups(groups=[])
        if self._client is not None:
            unique_names, appearances = self._collect_appearances(result)
            if unique_names:
                llm_groups = self._merge_via_llm(unique_names, appearances)

        final_groups = AliasGroups(groups=[*base_groups.groups, *llm_groups.groups])
        alias_map = self._build_alias_map(final_groups)
        return AliasingResult(alias_groups=final_groups, alias_map=alias_map)

    @staticmethod
    def _extract_from_entity_clues(result: PipelineResult) -> AliasGroups:
        groups: dict[str, set[str]] = defaultdict(set)
        entities = result.get(EntityClue)
        if not entities:
            return AliasGroups(groups=[])

        for clue in entities:
            canonical = clue.name
            groups[canonical].add(canonical)
            groups[canonical].update(clue.aliases_in_scene)

        return AliasGroups(
            groups=[
                AliasGroup(canonical=name, aliases=sorted(list(aliases)))
                for name, aliases in groups.items()
            ]
        )

    def _collect_appearances(
        self, result: PipelineResult
    ) -> tuple[list[str], dict[str, list[int]]]:
        """Collect name co-occurrences from Act/ToM clues"""
        unique_names: set[str] = set()
        appearances: dict[str, set[int]] = defaultdict(set)

        for clue in [*result.get(ActClue), *result.get(ToMClue)]:
            pair = clue.pair
            unique_names.update(pair)
            appearances[pair[0]].add(clue.scene)
            appearances[pair[1]].add(clue.scene)

        return sorted(unique_names), {k: sorted(v) for k, v in appearances.items()}

    def _merge_via_llm(
        self, unique_names: list[str], appearances: dict[str, list[int]]
    ) -> AliasGroups:
        """Merge aliases via LLM co-occurrence analysis"""
        assert self._client

        resp = self._client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": self._build_prompt(unique_names, appearances)}],
                }
            ],
            config=types.GenerateContentConfig(
                system_instruction=self._SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=AliasGroups,
            ),
        )

        raw = getattr(resp, "parsed", None)
        if raw is not None:
            return parse_model(AliasGroups, raw)
        if resp.text is not None:
            return parse_model(AliasGroups, resp.text)
        raise RuntimeError("Alias generation returned no usable payload.")

    @staticmethod
    def _build_alias_map(groups: AliasGroups) -> dict[str, str]:
        """Build {alias: canonical} mapping"""
        mp: dict[str, str] = {}
        for g in groups.groups:
            for a in g.aliases:
                mp[a] = g.canonical
        return mp

    @staticmethod
    def _build_prompt(
        unique_names: list[str], appearances: dict[str, list[int]]
    ) -> str:
        return json.dumps(
            {"UNIQUE_NAMES": unique_names, "APPEARANCES": appearances, "HINTS": []},
            ensure_ascii=False,
            indent=2,
        )

    _SYSTEM_PROMPT = """
    You merge aliases into canonical person identities.
    Merge ONLY if at least TWO hold:
    (1) co-occurrence/interaction context,
    (2) explicit aka/title/codename,
    (3) consistent role/relationship pattern.
    Be conservative. No guessing.
    Return strict JSON: { "groups": [ { "canonical": "...", "aliases": [...] } ] }.
    """.strip()


def alias_name(n: str, amap: dict[str, str]) -> str:
    """Apply alias map to a single name"""
    low = n.lower()
    for k, v in amap.items():
        if low == k.lower():
            return v
    return n


def alias_pair(p: tuple[str, str], amap: dict[str, str]) -> tuple[str, str]:
    """Apply alias map to a pair"""
    return norm_pair(alias_name(p[0], amap), alias_name(p[1], amap))


__all__ = ["AliasResolver", "alias_name", "alias_pair"]
