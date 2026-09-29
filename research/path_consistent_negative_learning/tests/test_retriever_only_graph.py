from collections import deque
import random
import unittest

import networkx as nx

from research.path_consistent_negative_learning.retriever_only.graph import (
    MAX_INCIDENCE_DISTANCE,
    shortest_path,
)


def _legacy_shortest_path(
    graph: nx.Graph,
    source: str,
    target: str,
    *,
    variant_seed: int,
) -> tuple[str, ...] | None:
    """The pre-optimization target-pop BFS, retained only as a test oracle."""

    rng = random.Random(f"{variant_seed}|{source}|{target}")
    parents: dict[str, str | None] = {source: None}
    distances = {source: 0}
    queue: deque[str] = deque([source])
    while queue:
        current = queue.popleft()
        if current == target:
            break
        if distances[current] >= MAX_INCIDENCE_DISTANCE:
            continue
        neighbors = sorted(graph.neighbors(current))
        rng.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor in parents:
                continue
            parents[neighbor] = current
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    if target not in parents:
        return None
    path = []
    node: str | None = target
    while node is not None:
        path.append(node)
        node = parents[node]
    return tuple(reversed(path))


class RetrieverGraphTests(unittest.TestCase):
    def test_target_discovery_exit_matches_legacy_target_pop_paths(self) -> None:
        for graph_seed in range(3):
            integer_graph = nx.gnp_random_graph(30, 0.16, seed=graph_seed)
            graph = nx.relabel_nodes(
                integer_graph,
                {node: f"node-{node:02d}" for node in integer_graph},
            )
            nodes = sorted(graph)
            cache = {
                node: tuple(sorted(graph.neighbors(node))) for node in graph
            }
            for variant_seed in (2718, 3141, 5772):
                for source in nodes:
                    for target in nodes:
                        self.assertEqual(
                            shortest_path(
                                graph,
                                source,
                                target,
                                variant_seed=variant_seed,
                                neighbor_cache=cache,
                            ),
                            _legacy_shortest_path(
                                graph,
                                source,
                                target,
                                variant_seed=variant_seed,
                            ),
                        )


if __name__ == "__main__":
    unittest.main()
