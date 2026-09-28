from __future__ import annotations

import unittest

import torch

from research.path_consistent_negative_learning.retriever_only.prepared import (
    PreparedCandidates,
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

    def test_lambda_zero_masks_exact_path_consistent_negatives(self) -> None:
        weights = method_weights(self.data, method="ours", lambda_=0.0, seed=42)
        self.assertEqual(torch.where(weights == 0)[0].tolist(), [1, 6])

    def test_matched_random_uses_equal_count_per_query(self) -> None:
        weights = method_weights(
            self.data,
            method="matched_random",
            lambda_=0.25,
            seed=42,
        )
        for start, stop in ((0, 4), (4, 8)):
            self.assertEqual(int((weights[start:stop] == 0.25).sum()), 1)
            self.assertFalse(
                torch.any(
                    (weights[start:stop] == 0.25)
                    & self.data.path_consistent_mask[start:stop]
                )
            )


if __name__ == "__main__":
    unittest.main()
