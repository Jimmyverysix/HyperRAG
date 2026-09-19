import json
from pathlib import Path
import tempfile
import unittest

from research.path_consistent_negative_learning.aggregate_training import (
    aggregate_runs,
)
from research.path_consistent_negative_learning.strategies import STRATEGIES


class TrainingAggregationTests(unittest.TestCase):
    def test_gate_c_uses_paired_query_differences(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for strategy in STRATEGIES:
                for seed in (42, 43):
                    run_dir = root / strategy / f"seed_{seed}"
                    run_dir.mkdir(parents=True)
                    bonus = 0.2 if strategy == "strategy2_ignore" else 0.0
                    result = {
                        "strategy": strategy,
                        "seed": seed,
                        "training_seconds": 1.0,
                        "peak_cuda_memory_bytes": 100.0,
                        "metrics": {
                            "selected_path": {
                                "mrr": 0.5 + bonus,
                                "hits_at": {"10": 0.5 + bonus},
                                "recall_at": {"10": 0.5 + bonus},
                                "question_pr_auc": 0.6,
                            },
                            "all_shortest_paths": {
                                "mrr": 0.5 + bonus,
                                "hits_at": {"10": 0.5 + bonus},
                                "recall_at": {"10": 0.5 + bonus},
                                "question_pr_auc": 0.5 + bonus,
                            },
                            "answer_reach_at": {"10": 0.5 + bonus},
                        },
                    }
                    (run_dir / "result.json").write_text(
                        json.dumps(result), encoding="utf-8"
                    )
                    rows = []
                    for query in ("q1", "q2"):
                        rows.append(
                            {
                                "query_key": query,
                                "selected_mrr": 0.5 + bonus,
                                "selected_pr_auc": 0.6,
                                "selected_hits_10": 0.5 + bonus,
                                "selected_recall_10": 0.5 + bonus,
                                "all_shortest_mrr": 0.5 + bonus,
                                "all_shortest_pr_auc": 0.5 + bonus,
                                "all_shortest_hits_10": 0.5 + bonus,
                                "all_shortest_recall_10": 0.5 + bonus,
                                "answer_reach_10": 0.5 + bonus,
                            }
                        )
                    (run_dir / "query_metrics.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows),
                        encoding="utf-8",
                    )
            report = aggregate_runs(root)
        self.assertTrue(report["gate_c"]["proceed_to_gate_d"])
        self.assertGreater(
            report["gate_c"]["strategy2_vs_strategy1_primary_ci_low"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
