from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence, Type

import networkx as nx
from pydantic import BaseModel

from framework.processor import Processor
from framework.result import PipelineResult
from schema import BaseClue

if TYPE_CHECKING:
    from framework.pipeline import PipelineConfig


@dataclass(slots=True)
class Edge:
    src: int
    dst: int
    weight: float


class TemporalResult(BaseModel):
    fabula_rank: dict[int, int]


class FabulaReconstructor:
    """Reconstruct chronological segment order from heterogeneous clues."""

    def reconstruct(
        self,
        segments: Iterable[int],
        clues: Sequence[BaseClue],
        input_metadata: dict[int, dict],
    ) -> dict[int, int]:
        nodes = list(dict.fromkeys(int(s) for s in segments))
        edges: list[Edge] = []

        for segment_id in nodes:
            refs = input_metadata.get(segment_id, {}).get(
                "referenced_segments", []
            )
            for ref in refs:
                edges.append(Edge(int(ref), segment_id, weight=1.0))

        for clue in clues:
            if getattr(clue, "clue_type", "") == "temporal":
                for ref in getattr(clue, "referenced_segments", []):
                    edges.append(Edge(int(ref), int(clue.segment), weight=0.8))

        for clue in clues:
            if getattr(clue, "clue_type", "") == "act":
                for ref in getattr(clue, "referenced_segments", []):
                    edges.append(Edge(int(clue.segment), int(ref), weight=0.6))

        order = self._topological_sort(nodes, edges)
        return {segment_id: idx for idx, segment_id in enumerate(order)}

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


class TemporalReconstructor(Processor):
    """
    Reconstruct chronological segment order (fabula) from narrative order (syuzhet).

    Requirements: None — relies on graph-based topological sorting only.

    Input: All clues (temporal references, consequence chains) plus metadata.
    Output: TemporalResult containing fabula_rank mapping.
    """

    def __init__(self) -> None:
        self._engine = FabulaReconstructor()

    def configure(self, config: "PipelineConfig") -> None:  # noqa: D401
        _ = config

    def __call__(self, result: PipelineResult) -> TemporalResult:
        segments = [int(item["segment"]) for item in result.segments]
        metadata = result.get_context(
            "segment_metadata", namespace="framework", default={}
        )
        fabula = self._engine.reconstruct(
            segments,
            result.all_clues,
            metadata if isinstance(metadata, dict) else {},
        )
        return TemporalResult(fabula_rank=fabula)

    def checkpoint_id(self) -> str:
        cls = self.__class__
        return f"{cls.__module__}.{cls.__qualname__}"

    @property
    def result_type(self) -> Type[TemporalResult]:
        return TemporalResult


__all__ = ["TemporalReconstructor", "FabulaReconstructor", "TemporalResult"]
