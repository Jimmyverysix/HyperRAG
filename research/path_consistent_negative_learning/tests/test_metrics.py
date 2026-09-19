import unittest

from research.path_consistent_negative_learning.metrics import (
    evaluate_query,
    evaluate_retrieval,
    precision_recall_auc,
)


class RetrievalMetricsTests(unittest.TestCase):
    def test_query_metrics_use_first_positive_and_all_positive_labels(self) -> None:
        result = evaluate_query(
            [0.9, 0.8, 0.7],
            [0, 1, 1],
            ks=(1, 2, 3),
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.reciprocal_rank, 0.5)
        self.assertEqual(result.hits_at, {1: 0.0, 2: 1.0, 3: 1.0})
        self.assertEqual(result.recall_at, {1: 0.0, 2: 0.5, 3: 1.0})
        self.assertAlmostEqual(result.pr_auc, (0.5 + 2.0 / 3.0) / 2.0)

    def test_pr_auc_is_invariant_to_tie_order(self) -> None:
        first = precision_recall_auc([0.9, 0.9, 0.1], [1, 0, 1])
        second = precision_recall_auc([0.9, 0.9, 0.1], [0, 1, 1])
        self.assertAlmostEqual(first, second)
        self.assertAlmostEqual(first, 7.0 / 12.0)

    def test_aggregate_exposes_question_and_candidate_metrics(self) -> None:
        result = evaluate_retrieval(
            {
                "q1": ([0.9, 0.8, 0.7], [0, 1, 1]),
                "q2": ([0.9, 0.1], [1, 0]),
            },
            ks=(1, 2),
        )
        self.assertEqual(result.query_count, 2)
        self.assertEqual(result.evaluated_query_count, 2)
        self.assertEqual(result.candidate_count, 5)
        self.assertEqual(result.positive_count, 3)
        self.assertAlmostEqual(result.mrr, 0.75)
        self.assertEqual(result.hits_at, {1: 0.5, 2: 1.0})
        self.assertEqual(result.recall_at, {1: 0.5, 2: 0.75})
        self.assertEqual(result.micro_recall_at, {1: 1.0 / 3.0, 2: 2.0 / 3.0})
        self.assertAlmostEqual(
            result.question_pr_auc,
            ((0.5 + 2.0 / 3.0) / 2.0 + 1.0) / 2.0,
        )
        expected_candidate_auc = precision_recall_auc(
            [0.9, 0.8, 0.7, 0.9, 0.1],
            [0, 1, 1, 1, 0],
        )
        self.assertAlmostEqual(result.candidate_pr_auc, expected_candidate_auc)

    def test_no_positive_policies_are_explicit(self) -> None:
        queries = {
            "positive": ([0.8, 0.2], [1, 0]),
            "no-positive": ([0.7, 0.1], [0, 0]),
        }
        skipped = evaluate_retrieval(queries, ks=(1,), no_positive="skip")
        self.assertEqual(skipped.evaluated_query_count, 1)
        self.assertEqual(skipped.skipped_query_count, 1)
        self.assertIsNone(skipped.per_query["no-positive"])
        self.assertEqual(skipped.mrr, 1.0)

        zeroed = evaluate_retrieval(queries, ks=(1,), no_positive="zero")
        self.assertEqual(zeroed.evaluated_query_count, 2)
        self.assertEqual(zeroed.skipped_query_count, 0)
        self.assertEqual(zeroed.mrr, 0.5)
        self.assertEqual(zeroed.question_pr_auc, 0.5)

        with self.assertRaisesRegex(ValueError, "no positive"):
            evaluate_retrieval(queries, ks=(1,), no_positive="error")

    def test_all_skipped_queries_return_well_defined_zero_aggregates(self) -> None:
        result = evaluate_retrieval(
            {"q": ([0.5, 0.4], [0, 0])},
            ks=(1, 10),
            no_positive="skip",
        )
        self.assertEqual(result.evaluated_query_count, 0)
        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(result.mrr, 0.0)
        self.assertEqual(result.hits_at, {1: 0.0, 10: 0.0})
        self.assertEqual(result.candidate_pr_auc, 0.0)

    def test_invalid_candidate_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            evaluate_query([0.2], [0, 1])
        with self.assertRaisesRegex(ValueError, "binary"):
            evaluate_query([0.2], [2])
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_query([float("nan")], [1])
        with self.assertRaisesRegex(ValueError, "positive integer"):
            evaluate_query([0.2], [1], ks=(0,))


if __name__ == "__main__":
    unittest.main()
