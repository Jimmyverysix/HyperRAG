from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research.path_consistent_negative_learning.retriever_only.aggregation import (
    aggregate_main_test,
    aggregate_path_sensitivity,
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

    def test_main_aggregate_includes_domain_level_paired_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            for method, score in (
                ("baseline", 0.1),
                ("matched_random", 0.2),
                ("ours", 0.3),
            ):
                for seed in (42, 43, 44, 45, 46):
                    path = (
                        run_root
                        / "test"
                        / "toy"
                        / method
                        / f"seed_{seed}"
                        / "test_scores.report.json"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps(
                            {
                                "split": "test",
                                "method": method,
                                "seed": seed,
                                "queries": [
                                    {
                                        "query_key": "q",
                                        "reciprocal_rank": score,
                                        "answer_reach_10": score,
                                        "answer_reach_5": score,
                                    }
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
            result = aggregate_main_test(run_root, ("toy",))
        self.assertEqual(len(result["domain_comparisons"]), 2)
        first = result["domain_comparisons"][0]
        self.assertEqual(first["domain"], "toy")
        self.assertEqual(first["metric"], "APC-MRR")
        self.assertAlmostEqual(first["difference_percentage_points"], 20.0)

    def test_path_sensitivity_averages_seeds_before_variants(self) -> None:
        variant_scores = {
            2718: (0.0, 1.0),
            3141: (0.2, 0.8),
            5772: (0.4, 0.6),
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            for method in ("baseline", "ours"):
                for variant, scores in variant_scores.items():
                    for seed, score in zip((42, 43), scores):
                        path = (
                            run_root
                            / "sensitivity"
                            / "toy"
                            / f"variant_{variant}"
                            / f"seed_{seed}"
                            / method
                            / "test_scores.report.json"
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(
                            json.dumps(
                                {
                                    "evaluation_subset": (
                                        "multiple_equal_shortest_paths"
                                    ),
                                    "method": method,
                                    "seed": seed,
                                    "query_count": 5,
                                    "metrics": {
                                        "answer_path_mrr": score,
                                        "answer_reach_10": score,
                                    },
                                }
                            ),
                            encoding="utf-8",
                        )
            result = aggregate_path_sensitivity(
                run_root,
                ("toy",),
                variants=(2718, 3141, 5772),
                seeds=(42, 43),
            )
        baseline = result["domains"][0]
        self.assertAlmostEqual(
            baseline["answer_path_mrr"]["standard_deviation"],
            0.0,
        )
        self.assertEqual(
            baseline["answer_path_mrr"]["per_variant_seed_average"],
            {"2718": 0.5, "3141": 0.5, "5772": 0.5},
        )
        fields, rows = _csv_rows("sensitivity", result)
        self.assertIn("variant_2718_seed_average", fields)
        self.assertEqual(list(rows)[0]["variant_2718_seed_average"], 0.5)


if __name__ == "__main__":
    unittest.main()
