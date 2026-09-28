from __future__ import annotations

import unittest

import numpy as np

from research.path_consistent_negative_learning.retriever_only.aggregation import (
    average_query_metrics,
    paired_equal_domain_bootstrap,
)
from research.path_consistent_negative_learning.scripts.aggregate_retriever_results import (
    _csv_rows,
)


class RetrieverAggregationTests(unittest.TestCase):
    def test_seed_average_keeps_query_as_unit(self) -> None:
        reports = [
            {
                "queries": [
                    {
                        "query_key": "q",
                        "reciprocal_rank": value,
                        "answer_reach_10": 1.0,
                        "answer_reach_5": 0.0,
                    }
                ]
            }
            for value in (0.2, 0.4)
        ]
        result = average_query_metrics(reports)
        self.assertAlmostEqual(result["q"]["reciprocal_rank"], 0.3)

    def test_bootstrap_uses_equal_domain_macro(self) -> None:
        result = paired_equal_domain_bootstrap(
            {
                "large": np.ones(100),
                "small": np.asarray([-1.0]),
            },
            resamples=100,
            seed=7,
            batch_size=10,
        )
        self.assertAlmostEqual(result["difference"], 0.0)
        self.assertAlmostEqual(result["ci_low"], 0.0)
        self.assertAlmostEqual(result["ci_high"], 0.0)

    def test_main_csv_is_tidy_and_includes_macro_rows(self) -> None:
        metrics = {
            "reciprocal_rank": 0.2,
            "answer_reach_10": 0.3,
            "answer_reach_5": 0.1,
        }
        _, row_values = _csv_rows(
            "main-test",
            {
                "domains": [
                    {
                        "domain": "toy",
                        "baseline": metrics,
                        "matched_random": metrics,
                        "ours": metrics,
                    }
                ],
                "equal_domain_macro": {
                    "baseline": metrics,
                    "matched_random": metrics,
                    "ours": metrics,
                },
            },
        )
        rows = list(row_values)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[-1]["scope"], "equal_domain_macro")
        self.assertEqual(rows[-1]["answer_path_mrr"], 0.2)


if __name__ == "__main__":
    unittest.main()
