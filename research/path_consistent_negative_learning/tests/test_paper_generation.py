import csv
import json
import tempfile
import unittest
from pathlib import Path

from research.path_consistent_negative_learning.scripts.generate_paper_results import (
    generate,
)


class PaperGenerationTests(unittest.TestCase):
    def test_generates_macros_and_tables_only_from_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            historical = root / "historical.json"
            audit = root / "audit.json"
            summary = root / "summary.json"
            table = root / "table.csv"
            selection = root / "selection.csv"
            historical.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "facts": {
                            "lambda_zero_selected_pr_auc_delta": -0.013,
                            "historical_transfer_lambda": 0.1,
                            "historical_transfer_answer_reach_delta": 0.048,
                            "historical_transfer_selected_pr_auc_delta": -0.002,
                        },
                    }
                ),
                encoding="utf-8",
            )
            domain_audit = {
                "domain": "toy",
                "query_count": 10,
                "sampled_conflict_rate": 0.1,
                "queries_with_sampled_conflict_any_seed_rate": 0.2,
            }
            audit.write_text(
                json.dumps(
                    {
                        "historical_totals_verified": True,
                        "overall": {
                            "query_count": 10,
                            "sampled_conflict_rate": 0.1,
                            "queries_with_sampled_conflict_any_seed_rate": 0.2,
                        },
                        "domains": [domain_audit],
                    }
                ),
                encoding="utf-8",
            )
            comparisons = {
                "dataset_specific_lambda_star_vs_baseline_lambda_1_answer_reach_10": {
                    "mean_difference": 0.03, "ci_low": 0.01, "ci_high": 0.05
                },
                "dataset_specific_lambda_star_vs_baseline_lambda_1_selected_pr_auc": {
                    "mean_difference": -0.004, "ci_low": -0.006, "ci_high": -0.002
                },
                "dataset_specific_lambda_star_vs_matched_random_at_lambda_star_answer_reach_10": {
                    "mean_difference": 0.02, "ci_low": 0.0, "ci_high": 0.04
                },
                "dataset_specific_lambda_star_vs_matched_random_at_lambda_star_selected_pr_auc": {
                    "mean_difference": -0.003, "ci_low": -0.005, "ci_high": -0.001
                },
            }
            summary.write_text(
                json.dumps(
                    {
                        "scope": "structured_proxy_only_not_official_hyperrag_qa",
                        "domains": {"toy": {}},
                        "macro_comparisons": comparisons,
                        "lambda_conflict_correlation": {"spearman_rho": -0.5, "p_value": 0.1},
                    }
                ),
                encoding="utf-8",
            )
            with table.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "dataset", "selected_lambda", "baseline_answer_reach_10_mean",
                        "tuned_answer_reach_10_mean", "tuned_minus_baseline_answer_reach_10",
                        "tuned_minus_baseline_selected_pr_auc", "mask_answer_reach_10_mean",
                        "random_answer_reach_10_mean", "all_positive_answer_reach_10_mean",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset": "toy", "selected_lambda": 0.25,
                        "baseline_answer_reach_10_mean": 0.7,
                        "tuned_answer_reach_10_mean": 0.73,
                        "tuned_minus_baseline_answer_reach_10": 0.03,
                        "tuned_minus_baseline_selected_pr_auc": -0.004,
                        "mask_answer_reach_10_mean": 0.72,
                        "random_answer_reach_10_mean": 0.71,
                        "all_positive_answer_reach_10_mean": 0.69,
                    }
                )
            with selection.open("w", encoding="utf-8", newline="") as handle:
                fieldnames = (
                    "dataset", "best_lambda", "lambda_0", "lambda_0.1",
                    "lambda_0.25", "lambda_0.5", "lambda_0.75", "lambda_1",
                )
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({column: 0.7 for column in fieldnames} | {"dataset": "toy", "best_lambda": 0.25})

            output = root / "generated"
            manifest = generate(historical, audit, summary, table, selection, output)

            results = (output / "results.tex").read_text(encoding="utf-8")
            self.assertIn(r"\newcommand{\TunedReachDelta}{3.00}", results)
            self.assertIn(r"\newcommand{\TunedVsRandomPRDelta}{-0.30}", results)
            self.assertIn(r"\newcommand{\InteriorSelectedDomainCount}{1}", results)
            self.assertIn(r"\newcommand{\PRGuardrailViolationCount}{0}", results)
            self.assertIn("toy", (output / "tables" / "main_proxy.tex").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_count"], 1)


if __name__ == "__main__":
    unittest.main()
