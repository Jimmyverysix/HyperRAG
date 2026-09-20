import unittest

from research.path_consistent_negative_learning.path_supervision.lambda_selection import (
    DEFAULT_LAMBDA_GRID,
    select_lambda,
)


def _result(lambda_value, seed, score, split="selection"):
    return {
        "domain": "toy",
        "strategy": "strategy2_weighted",
        "seed": seed,
        "evaluation_split": split,
        "disputed_negative_weight": lambda_value,
        "metrics": {
            "answer_reach_at": {"10": score},
            "selected_path": {"mrr": score, "question_pr_auc": score},
            "all_shortest_paths": {"mrr": score, "question_pr_auc": score},
        },
    }


class LambdaSelectionTests(unittest.TestCase):
    def test_uses_all_grid_points_and_shared_seeds(self) -> None:
        records = []
        for lambda_value in DEFAULT_LAMBDA_GRID:
            for seed in (42, 43):
                score = 0.8 - abs(lambda_value - 0.25)
                records.append(_result(lambda_value, seed, score))

        selected = select_lambda(records, selection_metric="answer_reach_10")

        self.assertEqual(selected["best_lambda"], 0.25)
        self.assertFalse(selected["test_metrics_accessed"])
        self.assertEqual(selected["seeds"], [42, 43])

    def test_rejects_test_results(self) -> None:
        records = [
            _result(lambda_value, 42, 0.8, split="test")
            for lambda_value in DEFAULT_LAMBDA_GRID
        ]
        with self.assertRaisesRegex(ValueError, "test leakage"):
            select_lambda(records, selection_metric="answer_reach_10")

    def test_exact_tie_prefers_weaker_intervention(self) -> None:
        records = [
            _result(lambda_value, 42, 0.9)
            for lambda_value in DEFAULT_LAMBDA_GRID
        ]
        selected = select_lambda(records, selection_metric="answer_reach_10")
        self.assertEqual(selected["best_lambda"], 1.0)


if __name__ == "__main__":
    unittest.main()
