from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx

from clues.act import ActClue
from clues.temporal import TemporalClue
from schema import BaseSignal


@dataclass(slots=True)
class Edge:
    src: int
    dst: int
    weight: float


class FabulaReconstructor:
    """Reconstruct chronological scene order from heterogeneous signals."""

    def reconstruct(
        self,
        scenes: Iterable[int],
        signals: Sequence[BaseSignal],
        input_metadata: dict[int, dict],
    ) -> dict[int, int]:
        nodes = list(dict.fromkeys(int(s) for s in scenes))
        edges: list[Edge] = []

        for scene_id in nodes:
            refs = input_metadata.get(scene_id, {}).get("references_scenes", [])
            for ref in refs:
                edges.append(Edge(int(ref), scene_id, weight=1.0))

        for signal in signals:
            if isinstance(signal, TemporalClue):
                for ref in signal.references_scenes:
                    edges.append(Edge(int(ref), int(signal.scene), weight=0.8))

        for signal in signals:
            if isinstance(signal, ActClue):
                for ref in signal.axes.consequence_refs:
                    edges.append(Edge(int(signal.scene), int(ref), weight=0.6))

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
            order = nodes[:]  # fallback to input order if still cyclic

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

            # Remove the lowest-weight edge in the detected cycle
            min_edge = min(
                cycle,
                key=lambda e: graph.edges[e[0], e[1]].get("weight", 0.0),
            )
            graph.remove_edge(min_edge[0], min_edge[1])
            if nx.is_directed_acyclic_graph(graph):
                return
