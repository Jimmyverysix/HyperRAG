from __future__ import annotations

import random
import unittest

import networkx as nx

from research.path_consistent_negative_learning.retriever_only.candidates import (
    enumerate_retrieval_candidates,
    path_consistent_transitions,
    sample_negative_transitions,
)
from research.path_consistent_negative_learning.retriever_only.metrics import (
    answer_path_metrics,
    rank_transitions,
)


def add_transition(graph: nx.Graph, head: str, edge: str, tail: str) -> None:
    graph.add_edge(head, edge)
    graph.add_edge(edge, tail)


class PathConsistentCandidateTests(unittest.TestCase):
    def test_equal_shortest_path_is_recognized(self) -> None:
        graph = nx.Graph()
        add_transition(graph, "s", "H:1", "b")
        add_transition(graph, "b", "H:2", "a")
        add_transition(graph, "s", "H:3", "c")
        add_transition(graph, "c", "H:4", "a")
        candidates = (("s", "H:3", "c"), ("c", "H:4", "a"))
        self.assertEqual(
            set(path_consistent_transitions(graph, "s", ("a",), candidates)),
            set(candidates),
        )

    def test_non_shortest_and_local_only_paths_are_rejected(self) -> None:
        graph = nx.Graph()
        add_transition(graph, "s", "H:0", "a")
        add_transition(graph, "s", "H:1", "v")
        add_transition(graph, "v", "H:2", "u")
        add_transition(graph, "u", "H:3", "a")
        candidate = ("v", "H:2", "u")
        self.assertEqual(
            path_consistent_transitions(graph, "s", ("a",), (candidate,)),
            (),
        )

    def test_sampler_is_reproducible_and_equal_count_when_pool_allows(self) -> None:
        graph = nx.Graph()
        for entity in ("a", "b", "c", "d"):
            graph.add_edge(entity, "H:x")
        positives = (("a", "H:x", "b"),)
        first = sample_negative_transitions(
            graph, positives, rng=random.Random(42), count=1
        )
        second = sample_negative_transitions(
            graph, positives, rng=random.Random(42), count=1
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertNotIn(first[0], {positives[0], ("b", "H:x", "a")})

    def test_exhaustive_candidates_do_not_need_answers(self) -> None:
        graph = nx.Graph()
        add_transition(graph, "s", "H:1", "b")
        add_transition(graph, "b", "H:2", "a")
        candidates = enumerate_retrieval_candidates(graph, "s", maximum_hops=2)
        self.assertEqual(
            tuple(value.transition for value in candidates),
            (("s", "H:1", "b"), ("b", "H:2", "a")),
        )

    def test_answer_path_rank_uses_directed_connectivity(self) -> None:
        transitions = (
            ("x", "H:0", "a"),
            ("s", "H:1", "b"),
            ("b", "H:2", "a"),
        )
        ranked = rank_transitions(transitions, (0.9, 0.8, 0.7))
        metrics = answer_path_metrics(ranked, "s", ("a",))
        self.assertEqual(metrics.first_answer_rank, 3)
        self.assertAlmostEqual(metrics.reciprocal_rank, 1 / 3)
        self.assertEqual(metrics.reach_at_5, 1.0)


if __name__ == "__main__":
    unittest.main()
