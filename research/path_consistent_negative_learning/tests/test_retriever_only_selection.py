from __future__ import annotations

import unittest

from research.path_consistent_negative_learning.retriever_only.selection import (
    LAMBDA_GRID,
    SEEDS,
    select_domain_lambda,
)


def reports(scores: dict[float, float]) -> list[dict]:
    return [
        {
            "domain": "toy",
            "split": "valid",
            "lambda": lambda_,
            "seed": seed,
            "metrics": {"answer_path_mrr": score},
        }
        for lambda_, score in scores.items()
        for seed in SEEDS
    ]


class RetrieverLambdaSelectionTests(unittest.TestCase):
    def test_maximizes_validation_mrr(self) -> None:
        values = {value: 0.4 for value in LAMBDA_GRID}
        values[0.25] = 0.5
        selection = select_domain_lambda(reports(values))
        self.assertEqual(selection["lambda"], 0.25)

    def test_exact_tie_chooses_larger_lambda(self) -> None:
        values = {value: 0.4 for value in LAMBDA_GRID}
        values[0.0] = 0.5
        values[0.1] = 0.5
        selection = select_domain_lambda(reports(values))
        self.assertEqual(selection["lambda"], 0.1)

    def test_incomplete_seed_grid_is_rejected(self) -> None:
        values = reports({value: 0.4 for value in LAMBDA_GRID})
        values.pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            select_domain_lambda(values)


if __name__ == "__main__":
    unittest.main()
