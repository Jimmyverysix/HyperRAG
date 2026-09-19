import unittest
from types import SimpleNamespace

import torch

from research.path_consistent_negative_learning.metrics import evaluate_retrieval
from research.path_consistent_negative_learning.structured_data import TEST_SPLIT
from research.path_consistent_negative_learning.train_structured import (
    _query_metric_rows,
    weighted_binary_cross_entropy,
)


class QueryMetricRowsTests(unittest.TestCase):
    def test_empty_test_query_is_skipped(self):
        data = SimpleNamespace(
            query_keys=("empty", "kept"),
            query_splits=torch.tensor([TEST_SPLIT, TEST_SPLIT]),
            query_offsets=torch.tensor([0, 0, 1]),
        )
        metrics = evaluate_retrieval(
            {"kept": ([0.8], [1])},
            ks=(1, 3),
            no_positive="skip",
        )

        rows = _query_metric_rows(
            data,
            torch.tensor([0.8]),
            metrics,
            metrics,
            {3: {"kept": 1.0}},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query_key"], "kept")
        self.assertEqual(rows[0]["answer_reach_3"], 1.0)

    def test_weighted_bce_uses_weight_sum_and_zero_weight_has_no_gradient(self):
        logits = torch.tensor([0.0, 1.0], requires_grad=True)
        labels = torch.tensor([0.0, 0.0])
        weights = torch.tensor([1.0, 0.0])
        loss = weighted_binary_cross_entropy(logits, labels, weights)
        self.assertAlmostEqual(
            float(loss.detach()),
            float(torch.log(torch.tensor(2.0))),
            places=6,
        )
        loss.backward()
        self.assertAlmostEqual(float(logits.grad[1]), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
