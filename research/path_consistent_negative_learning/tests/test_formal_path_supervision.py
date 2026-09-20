import unittest

import networkx as nx
import torch
from torch.nn import functional as F

from research.path_consistent_negative_learning.distances import (
    distances_to_any,
    makes_local_answer_progress,
)
from research.path_consistent_negative_learning.path_supervision.path_consistency import (
    build_distance_index,
    find_path_consistent_negatives,
    is_path_consistent,
)
from research.path_consistent_negative_learning.path_supervision.weighted_loss import (
    path_consistent_weak_negative_loss,
)


def _add_transition(graph, head, fact, tail):
    graph.add_edge(head, fact)
    graph.add_edge(fact, tail)
    return head, fact, tail


class FormalPathConsistencyTests(unittest.TestCase):
    def test_equal_shortest_branch_is_path_consistent(self) -> None:
        graph = nx.DiGraph()
        chosen = _add_transition(graph, "s", "f1", "x")
        alternative = _add_transition(graph, "s", "f2", "y")
        _add_transition(graph, "x", "f3", "a")
        _add_transition(graph, "y", "f4", "a")
        index = build_distance_index(graph, ["s"], ["a"])

        self.assertTrue(is_path_consistent(chosen, index))
        self.assertEqual(
            find_path_consistent_negatives([alternative], index), (True,)
        )

    def test_longer_answer_reaching_branch_is_not_path_consistent(self) -> None:
        graph = nx.DiGraph()
        _add_transition(graph, "s", "f1", "x")
        _add_transition(graph, "x", "f2", "a")
        detour = _add_transition(graph, "s", "f3", "y")
        _add_transition(graph, "y", "f4", "z")
        _add_transition(graph, "z", "f5", "a")
        index = build_distance_index(graph, ["s"], ["a"])

        self.assertFalse(is_path_consistent(detour, index))

    def test_local_progress_can_hold_after_a_non_shortest_prefix(self) -> None:
        graph = nx.DiGraph()
        _add_transition(graph, "s", "short1", "x")
        _add_transition(graph, "x", "short2", "a")
        _add_transition(graph, "s", "detour1", "p")
        _add_transition(graph, "p", "detour2", "q")
        local_only = _add_transition(graph, "q", "detour3", "a")
        index = build_distance_index(graph, ["s"], ["a"])

        self.assertTrue(
            makes_local_answer_progress(graph, local_only, distances_to_any(graph, ["a"]))
        )
        self.assertFalse(is_path_consistent(local_only, index))


class WeightedObjectiveEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = torch.tensor([-1.1, 0.3, 1.7, -0.4], requires_grad=True)
        self.labels = torch.tensor([0.0, 0.0, 1.0, 0.0])
        self.disputed = torch.tensor([False, True, False, True])

    def test_lambda_one_equals_baseline_mean_bce(self) -> None:
        actual = path_consistent_weak_negative_loss(
            self.logits, self.labels, self.disputed, 1.0
        )
        expected = F.binary_cross_entropy_with_logits(self.logits, self.labels)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)

    def test_lambda_zero_equals_deleting_disputed_negatives(self) -> None:
        actual = path_consistent_weak_negative_loss(
            self.logits, self.labels, self.disputed, 0.0
        )
        keep = ~self.disputed
        expected = F.binary_cross_entropy_with_logits(
            self.logits[keep], self.labels[keep]
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)

        actual.backward()
        self.assertEqual(self.logits.grad[self.disputed].abs().sum().item(), 0.0)

    def test_labels_are_not_modified(self) -> None:
        before = self.labels.clone()
        path_consistent_weak_negative_loss(
            self.logits, self.labels, self.disputed, 0.25
        )
        torch.testing.assert_close(self.labels, before)


if __name__ == "__main__":
    unittest.main()
