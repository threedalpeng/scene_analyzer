from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import networkx as nx

from framework.processor import Processor
from framework.result import PipelineResult
from processors.types import TemporalResult
from schema import BaseClue

if TYPE_CHECKING:
    from framework.pipeline import PipelineConfig


@dataclass(slots=True)
class Edge:
    src: int
    dst: int
    weight: float


class FabulaReconstructor(Processor):
    """Reconstruct chronological scene order from heterogeneous clues."""

    def reconstruct(
        self,
        scenes: Iterable[int],
        clues: Sequence[BaseClue],
        input_metadata: dict[int, dict],
    ) -> dict[int, int]:
        nodes = list(dict.fromkeys(int(s) for s in scenes))
        edges: list[Edge] = []

        for scene_id in nodes:
            refs = input_metadata.get(scene_id, {}).get("references_scenes", [])
            for ref in refs:
                edges.append(Edge(int(ref), scene_id, weight=1.0))

        for clue in clues:
            if getattr(clue, "clue_type", "") == "temporal":
                for ref in getattr(clue, "references_scenes", []):
                    edges.append(Edge(int(ref), int(clue.scene), weight=0.8))

        for clue in clues:
            if getattr(clue, "clue_type", "") == "act":
                axes = getattr(clue, "axes", None)
                consequence_refs = getattr(axes, "consequence_refs", []) if axes else []
                for ref in consequence_refs:
                    edges.append(Edge(int(clue.scene), int(ref), weight=0.6))

        order = self._topological_sort(nodes, edges)
        return {scene_id: idx for idx, scene_id in enumerate(order)}

    @staticmethod
    def _topological_sort(nodes: list[int], edges: Sequence[Edge]) -> list[int]:
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for edge in edges:
            graph.add_edge(edge.src, edge.dst, weight=edge.weight)

        if not nx.is_directed_acyclic_graph(graph):
            FabulaReconstructor._break_cycles(graph)

        try:
            order = list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            order = nodes[:]

        unknown_nodes = [n for n in nodes if n not in order]
        return order + unknown_nodes

    @staticmethod
    def _break_cycles(graph: nx.DiGraph) -> None:
        while True:
            try:
                cycle = nx.find_cycle(graph, orientation="original")
            except nx.NetworkXNoCycle:
                return

            if not cycle:
                return

            min_edge = min(
                cycle,
                key=lambda e: graph.edges[e[0], e[1]].get("weight", 0.0),
            )
            graph.remove_edge(min_edge[0], min_edge[1])
            if nx.is_directed_acyclic_graph(graph):
                return


class TemporalReconstructor:
    """
    Reconstruct chronological scene order (fabula) from narrative order (syuzhet).

    Requirements: None — relies on graph-based topological sorting only.

    Input: All clues (temporal references, consequence chains) plus metadata.
    Output: TemporalResult containing fabula_rank mapping.
    """

    def __init__(self) -> None:
        self._engine = FabulaReconstructor()

    def configure(self, config: "PipelineConfig") -> None:  # noqa: D401
        _ = config

    def __call__(self, result: PipelineResult) -> TemporalResult:
        scenes = [int(item["scene"]) for item in result.scenes]
        metadata = result.metadata or {}
        fabula = self._engine.reconstruct(scenes, result.all_clues, metadata)
        return TemporalResult(fabula_rank=fabula)


__all__ = ["TemporalReconstructor", "FabulaReconstructor"]
