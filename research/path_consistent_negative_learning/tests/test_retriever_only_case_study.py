from __future__ import annotations

import unittest

import networkx as nx

from research.path_consistent_negative_learning.retriever_only.case_study import (
    witness_shortest_path,
)


class CaseStudyWitnessTests(unittest.TestCase):
    def test_witness_contains_the_requested_transition(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from(
            [
                ("s", "H:1"),
                ("H:1", "x"),
                ("x", "H:2"),
                ("H:2", "a"),
                ("s", "H:3"),
                ("H:3", "y"),
                ("y", "H:4"),
                ("H:4", "a"),
            ]
        )
        transition = ("s", "H:3", "y")
        path = witness_shortest_path(graph, "s", ("a",), transition)
        self.assertEqual(path, ("s", "H:3", "y", "H:4", "a"))

    def test_non_shortest_transition_has_no_witness(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from(
            [
                ("s", "H:0"),
                ("H:0", "a"),
                ("s", "H:1"),
                ("H:1", "x"),
                ("x", "H:2"),
                ("H:2", "a"),
            ]
        )
        self.assertIsNone(
            witness_shortest_path(graph, "s", ("a",), ("s", "H:1", "x"))
        )


if __name__ == "__main__":
    unittest.main()
