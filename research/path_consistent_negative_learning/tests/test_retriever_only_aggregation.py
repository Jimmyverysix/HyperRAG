from __future__ import annotations

import unittest

import numpy as np

from research.path_consistent_negative_learning.retriever_only.aggregation import (
    average_query_metrics,
    paired_equal_domain_bootstrap,
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


if __name__ == "__main__":
    unittest.main()
