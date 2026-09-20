import json
import tempfile
import unittest
from pathlib import Path

from research.path_consistent_negative_learning.scripts.aggregate_audit_distribution import (
    summarize_domain,
)


class WwwAuditAggregationTests(unittest.TestCase):
    def test_per_query_distribution_uses_seed_averages(self) -> None:
        records = [
            {
                "domain": "toy",
                "query_index": 0,
                "candidate_pool_size": 100,
                "disputed_candidate_count": 2,
                "disputed_candidate_rate": 0.02,
                "sampling": [
                    {
                        "sampled_count": 4,
                        "disputed_sampled_count": 0,
                        "disputed_sampled_rate": 0.0,
                    },
                    {
                        "sampled_count": 4,
                        "disputed_sampled_count": 2,
                        "disputed_sampled_rate": 0.5,
                    },
                ],
            },
            {
                "domain": "toy",
                "query_index": 1,
                "candidate_pool_size": 50,
                "disputed_candidate_count": 0,
                "disputed_candidate_rate": 0.0,
                "sampling": [
                    {
                        "sampled_count": 2,
                        "disputed_sampled_count": 0,
                        "disputed_sampled_rate": 0.0,
                    },
                    {
                        "sampled_count": 2,
                        "disputed_sampled_count": 0,
                        "disputed_sampled_rate": 0.0,
                    },
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "toy.audit.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
            summary, rows = summarize_domain(path)

        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["queries_with_candidate_conflict"], 1)
        self.assertEqual(summary["queries_with_sampled_conflict_any_seed"], 1)
        self.assertAlmostEqual(summary["sampled_conflict_rate"], 2 / 12)
        self.assertEqual(rows[0]["disputed_sampled_count_mean"], 1.0)
        self.assertEqual(
            summary["sampled_conflicts_per_query_seed_average"]["median"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
