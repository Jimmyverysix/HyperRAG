from __future__ import annotations

import unittest

from research.path_consistent_negative_learning.retriever_only.metrics import (
    answer_path_completion_rank_minimax,
    answer_path_metrics,
)


class AnswerPathCompletionMetricTests(unittest.TestCase):
    def test_incremental_and_minimax_definitions_are_equivalent(self) -> None:
        ranked = (
            ("s", "H:irrelevant", "x"),
            ("s", "H:first", "m"),
            ("x", "H:dead", "z"),
            ("m", "H:second", "a"),
        )
        incremental = answer_path_metrics(ranked, "s", ("a",))
        minimax = answer_path_completion_rank_minimax(ranked, "s", ("a",))
        self.assertEqual(incremental.first_answer_rank, 4)
        self.assertEqual(minimax, incremental.first_answer_rank)
        self.assertAlmostEqual(incremental.reciprocal_rank, 0.25)

    def test_candidate_pool_without_complete_path_has_zero_rr(self) -> None:
        ranked = (("s", "H:first", "m"), ("x", "H:second", "a"))
        incremental = answer_path_metrics(ranked, "s", ("a",))
        minimax = answer_path_completion_rank_minimax(ranked, "s", ("a",))
        self.assertIsNone(minimax)
        self.assertIsNone(incremental.first_answer_rank)
        self.assertEqual(incremental.reciprocal_rank, 0.0)


if __name__ == "__main__":
    unittest.main()
