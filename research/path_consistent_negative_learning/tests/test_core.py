import random
import unittest

import networkx as nx

from research.path_consistent_negative_learning.core import (
    answer_distances,
    audit_negative_pool,
    filter_path_consistent_negatives,
    is_path_consistent_transition,
    sample_negative_triplets,
)


def toy_hypergraph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("A", "<hyperedge>AB"),
            ("<hyperedge>AB", "B"),
            ("B", "<hyperedge>BC"),
            ("<hyperedge>BC", "C"),
            ("A", "<hyperedge>AD"),
            ("<hyperedge>AD", "D"),
            ("D", "<hyperedge>DC"),
            ("<hyperedge>DC", "C"),
            ("A", "<hyperedge>AE"),
            ("<hyperedge>AE", "E"),
            ("E", "<hyperedge>EF"),
            ("<hyperedge>EF", "F"),
        ]
    )
    return graph


class PathConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = toy_hypergraph()
        self.distances = answer_distances(self.graph, ["C"])

    def test_incidence_distance_uses_two_edges_per_logical_step(self) -> None:
        self.assertEqual(self.distances["A"], 4)
        self.assertEqual(self.distances["D"], 2)
        self.assertTrue(
            is_path_consistent_transition(("A", "<hyperedge>AD", "D"), self.distances)
        )
        self.assertFalse(
            is_path_consistent_transition(("D", "<hyperedge>AD", "A"), self.distances)
        )

    def test_filter_masks_each_optimal_progress_transition(self) -> None:
        negatives = [
            ("A", "<hyperedge>AD", "D"),
            ("A", "<hyperedge>AE", "E"),
            ("F", "<hyperedge>EF", "E"),
        ]
        retained, filtered = filter_path_consistent_negatives(negatives, self.distances)
        self.assertEqual(
            filtered,
            [("A", "<hyperedge>AD", "D"), ("F", "<hyperedge>EF", "E")],
        )
        self.assertEqual(retained, [("A", "<hyperedge>AE", "E")])

    def test_pool_audit_finds_path_consistent_unselected_transition(self) -> None:
        positives = {
            ("A", "<hyperedge>AB", "B"),
            ("B", "<hyperedge>BC", "C"),
        }
        audit = audit_negative_pool(self.graph, positives, self.distances)
        self.assertGreater(audit.candidate_pool_size, 0)
        self.assertGreaterEqual(audit.path_consistent_pool_size, 1)

    def test_ignore_policy_removes_without_replacement(self) -> None:
        positives = {
            ("A", "<hyperedge>AB", "B"),
            ("B", "<hyperedge>BC", "C"),
        }
        original = sample_negative_triplets(
            self.graph,
            positives,
            ["C"],
            num_samples=8,
            policy="original",
            rng=random.Random(7),
            distance_graph=self.graph,
        )
        ignored = sample_negative_triplets(
            self.graph,
            positives,
            ["C"],
            num_samples=8,
            policy="path-consistent-ignore",
            rng=random.Random(7),
            distance_graph=self.graph,
        )
        self.assertEqual(original.originally_sampled_count, ignored.originally_sampled_count)
        self.assertEqual(
            original.path_consistent_sampled_count,
            len(ignored.filtered),
        )
        self.assertEqual(set(original.negatives), set(ignored.negatives) | set(ignored.filtered))
        self.assertTrue(
            all(
                is_path_consistent_transition(triplet, self.distances)
                for triplet in ignored.filtered
            )
        )


if __name__ == "__main__":
    unittest.main()
