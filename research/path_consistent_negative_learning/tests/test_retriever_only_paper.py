from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research.path_consistent_negative_learning.retriever_only.paper import (
    generate_paper_artifacts,
)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class RetrieverPaperGenerationTests(unittest.TestCase):
    def test_generates_macros_and_strategy_123_tables(self) -> None:
        metrics = {
            "reciprocal_rank": 0.2,
            "answer_reach_10": 0.3,
            "answer_reach_5": 0.1,
        }
        comparison = {
            "difference_percentage_points": 0.5,
            "ci_low_percentage_points": 0.1,
            "ci_high_percentage_points": 0.9,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preflight = _write(
                root / "preflight.json",
                {
                    "domains": [
                        {
                            "domain": "toy",
                            "splits": {
                                "train": {
                                    "aligned_queries": 10,
                                    "retriever_eligible_queries": 9,
                                },
                                "valid": {
                                    "aligned_queries": 4,
                                    "retriever_eligible_queries": 4,
                                },
                                "test": {
                                    "aligned_queries": 5,
                                    "retriever_eligible_queries": 5,
                                },
                            },
                        }
                    ]
                },
            )
            validation = {
                str(value): {"mean_answer_path_mrr": 0.2}
                for value in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
            }
            selection = _write(
                root / "selection.json",
                {
                    "lambda_grid": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
                    "domains": {
                        "toy": {
                            "lambda": 0.5,
                            "validation": validation,
                        }
                    },
                },
            )
            prevalence = _write(
                root / "prevalence.json",
                {
                    "overall": {
                        "sampled_negative_count": 100,
                        "path_consistent_negative_count": 5,
                        "path_consistent_negative_rate": 0.05,
                        "equal_domain_affected_query_rate": 0.25,
                    },
                    "domains": [
                        {
                            "domain": "toy",
                            "sampled_negative_count": 100,
                            "path_consistent_negative_count": 5,
                            "path_consistent_negative_rate": 0.05,
                            "affected_query_rate_across_seeds": 0.25,
                        }
                    ],
                },
            )
            main = _write(
                root / "main.json",
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
                    "comparisons": {
                        "ours_minus_baseline": {
                            "reciprocal_rank": comparison,
                            "answer_reach_10": comparison,
                        },
                        "ours_minus_matched_random": {
                            "reciprocal_rank": comparison,
                            "answer_reach_10": comparison,
                        },
                    },
                },
            )
            sensitivity_rows = []
            for method in ("baseline", "ours"):
                sensitivity_rows.append(
                    {
                        "domain": "toy",
                        "method": method,
                        "query_count": 3,
                        "answer_path_mrr": {
                            "mean": 0.2,
                            "standard_deviation": 0.01,
                            "range": 0.03,
                        },
                        "answer_reach_10": {
                            "mean": 0.3,
                            "standard_deviation": 0.02,
                            "range": 0.04,
                        },
                    }
                )
            sensitivity = _write(
                root / "sensitivity.json", {"domains": sensitivity_rows}
            )
            output = root / "generated"
            generate_paper_artifacts(
                preflight,
                selection,
                prevalence,
                main,
                sensitivity,
                output,
            )
            macros = (output / "results.tex").read_text(encoding="utf-8")
            table = (output / "tables" / "main_retriever.tex").read_text(
                encoding="utf-8"
            )
            manifest = json.loads(
                (output / "generation_manifest.json").read_text(encoding="utf-8")
            )
        self.assertIn(r"\newcommand{\SoftSelectedDomainCount}{1}", macros)
        self.assertIn("策略1", table)
        self.assertIn("策略2", table)
        self.assertIn("策略3", table)
        self.assertIn("0.50", table)
        self.assertEqual(manifest["scope"], "formal_zero_llm_retriever_only")
        self.assertEqual(manifest["dataset_count"], 1)
        self.assertIn("tables/main_retriever.tex", manifest["outputs"])


if __name__ == "__main__":
    unittest.main()
