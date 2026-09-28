from __future__ import annotations

import unittest

import torch
from torch.nn import functional

from research.path_consistent_negative_learning.path_supervision.weighted_loss import (
    weighted_binary_cross_entropy,
)
from research.path_consistent_negative_learning.retriever_only.prepared import (
    PreparedCandidates,
    assemble_features,
    materialize_features,
    method_weights,
)


class PreparedWeightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = PreparedCandidates(
            domain="toy",
            split="train",
            seed=42,
            query_keys=["q1", "q2"],
            query_topics=["a", "b"],
            query_answers=[("c",), ("d",)],
            query_offsets=torch.tensor([0, 4, 8]),
            query_embedding_indices=torch.zeros(8, dtype=torch.long),
            head_embedding_indices=torch.zeros(8, dtype=torch.long),
            edge_embedding_indices=torch.zeros(8, dtype=torch.long),
            tail_embedding_indices=torch.zeros(8, dtype=torch.long),
            dde_features=torch.zeros((8, 30)),
            labels=torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.bool),
            path_consistent_mask=torch.tensor(
                [0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.bool
            ),
        )

    def test_baseline_and_lambda_one_are_identical(self) -> None:
        baseline = method_weights(
            self.data, method="baseline", lambda_=1.0, seed=42
        )
        ours = method_weights(self.data, method="ours", lambda_=1.0, seed=42)
        torch.testing.assert_close(baseline, ours)

    def test_lambda_one_matches_baseline_loss_and_gradient(self) -> None:
        baseline_logits = torch.tensor(
            [[-0.4], [0.2], [1.1], [-1.3]], requires_grad=True
        )
        ours_logits = baseline_logits.detach().clone().requires_grad_(True)
        labels = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        baseline_loss = functional.binary_cross_entropy_with_logits(
            baseline_logits, labels
        )
        ours_loss = weighted_binary_cross_entropy(
            ours_logits,
            labels,
            torch.ones_like(labels),
        )
        baseline_loss.backward()
        ours_loss.backward()
        torch.testing.assert_close(ours_loss, baseline_loss)
        torch.testing.assert_close(ours_logits.grad, baseline_logits.grad)

    def test_lambda_zero_masks_exact_path_consistent_negatives(self) -> None:
        weights = method_weights(self.data, method="ours", lambda_=0.0, seed=42)
        self.assertEqual(torch.where(weights == 0)[0].tolist(), [1, 6])

    def test_lambda_zero_matches_hard_masking_loss_and_gradient(self) -> None:
        weighted_logits = torch.tensor(
            [[-0.4], [0.2], [1.1], [-1.3]], requires_grad=True
        )
        masked_logits = weighted_logits.detach().clone().requires_grad_(True)
        labels = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        weights = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        weighted_loss = weighted_binary_cross_entropy(
            weighted_logits,
            labels,
            weights,
        )
        masked_loss = functional.binary_cross_entropy_with_logits(
            masked_logits[[0, 2]], labels[[0, 2]]
        )
        weighted_loss.backward()
        masked_loss.backward()
        torch.testing.assert_close(weighted_loss, masked_loss)
        torch.testing.assert_close(weighted_logits.grad, masked_logits.grad)

    def test_matched_random_uses_equal_count_per_query(self) -> None:
        weights = method_weights(
            self.data,
            method="matched_random",
            lambda_=0.25,
            seed=42,
        )
        for start, stop in ((0, 4), (4, 8)):
            self.assertEqual(int((weights[start:stop] == 0.25).sum()), 1)
            self.assertTrue(
                torch.all(
                    ~self.data.labels[start:stop][weights[start:stop] == 0.25]
                )
            )

    def test_matched_random_remains_defined_when_all_negatives_conflict(self) -> None:
        data = PreparedCandidates(
            domain="toy",
            split="train",
            seed=42,
            query_keys=["q"],
            query_topics=["a"],
            query_answers=[("b",)],
            query_offsets=torch.tensor([0, 2]),
            query_embedding_indices=torch.zeros(2, dtype=torch.long),
            head_embedding_indices=torch.zeros(2, dtype=torch.long),
            edge_embedding_indices=torch.zeros(2, dtype=torch.long),
            tail_embedding_indices=torch.zeros(2, dtype=torch.long),
            dde_features=torch.zeros((2, 30)),
            labels=torch.tensor([1, 0], dtype=torch.bool),
            path_consistent_mask=torch.tensor([0, 1], dtype=torch.bool),
        )
        weights = method_weights(
            data,
            method="matched_random",
            lambda_=0.0,
            seed=42,
        )
        torch.testing.assert_close(weights, torch.tensor([1.0, 0.0]))

    def test_chunked_feature_materialization_matches_direct_assembly(self) -> None:
        node_embeddings = torch.arange(3 * 1024, dtype=torch.float32).reshape(
            3,
            1024,
        )
        query_embeddings = torch.arange(
            2 * 1024,
            dtype=torch.float32,
        ).reshape(2, 1024)
        self.data.query_embedding_indices = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1]
        )
        self.data.head_embedding_indices = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        self.data.edge_embedding_indices = torch.tensor([1, 2, 0, 1, 2, 0, 1, 2])
        self.data.tail_embedding_indices = torch.tensor([2, 0, 1, 2, 0, 1, 2, 0])
        expected = assemble_features(
            self.data,
            node_embeddings,
            query_embeddings,
            torch.arange(8),
            device=torch.device("cpu"),
        )
        actual = materialize_features(
            self.data,
            node_embeddings,
            query_embeddings,
            device=torch.device("cpu"),
            chunk_size=3,
        )
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
