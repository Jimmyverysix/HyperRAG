from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from research.path_consistent_negative_learning.retriever_only.final_revision import (
    analyze_candidate_oracle,
    analyze_validation_lambdas,
)
from research.path_consistent_negative_learning.retriever_only.prepared import (
    PreparedCandidates,
)


class FinalRevisionAnalysisTests(unittest.TestCase):
    def test_validation_macro_uses_domains_equally_and_never_test(self) -> None:
        selection = {
            "lambda_grid": [0.0, 1.0],
            "test_metrics_accessed": False,
            "domains": {
                "large": {
                    "validation": {
                        "0.0": {"mean_answer_path_mrr": 0.4},
                        "1.0": {"mean_answer_path_mrr": 0.2},
                    }
                },
                "small": {
                    "validation": {
                        "0.0": {"mean_answer_path_mrr": 0.2},
                        "1.0": {"mean_answer_path_mrr": 0.1},
                    }
                },
            },
        }
        result = analyze_validation_lambdas(selection)
        self.assertEqual(result["best_global_lambda"], 0.0)
        self.assertAlmostEqual(
            result["rows"][0]["equal_domain_macro_apc_mrr"], 0.3
        )

    def test_validation_macro_rejects_post_test_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "pre-test"):
            analyze_validation_lambdas(
                {"test_metrics_accessed": True, "domains": {}, "lambda_grid": []}
            )

    def test_candidate_oracle_checks_complete_path_in_fixed_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = PreparedCandidates(
                domain="toy",
                split="test",
                seed=None,
                query_keys=["q1", "q2"],
                query_topics=["s1", "s2"],
                query_answers=[("a1",), ("a2",)],
                query_offsets=torch.tensor([0, 2, 3]),
                query_embedding_indices=torch.zeros(3, dtype=torch.long),
                head_embedding_indices=torch.zeros(3, dtype=torch.long),
                edge_embedding_indices=torch.zeros(3, dtype=torch.long),
                tail_embedding_indices=torch.zeros(3, dtype=torch.long),
                dde_features=torch.zeros((3, 30)),
                transitions=[
                    ("s1", "H:1", "m1"),
                    ("m1", "H:2", "a1"),
                    ("x", "H:3", "a2"),
                ],
                query_multiple_shortest=[False, False],
            )
            data.save(root / "prepared" / "eval" / "toy" / "test.pt")
            result = analyze_candidate_oracle(root, ["toy"])
        self.assertEqual(result["overall_query_count"], 2)
        self.assertEqual(result["overall_reachable_query_count"], 1)
        self.assertEqual(
            result["equal_domain_macro_candidate_oracle_reach"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
