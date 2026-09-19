import unittest

from research.path_consistent_negative_learning.reporting import aggregate_summaries


def summary(domain, candidate, disputed_candidate, sampled_by_seed):
    return {
        "domain": domain,
        "query_count": 10,
        "candidate_pool_size": candidate,
        "disputed_candidate_count": disputed_candidate,
        "sampling_by_seed": [
            {
                "seed": seed,
                "sampled_count": sampled,
                "disputed_sampled_count": disputed,
            }
            for seed, sampled, disputed in sampled_by_seed
        ],
    }


class AuditReportingTests(unittest.TestCase):
    def test_gate_continues_when_one_domain_crosses_threshold(self):
        report = aggregate_summaries(
            [
                summary("large", 10000, 20, [(1, 10000, 20), (2, 10000, 20)]),
                summary("small", 100, 2, [(1, 100, 2), (2, 100, 2)]),
            ]
        )
        self.assertLess(report["gate_b"]["global_sampled_rate"], 0.01)
        self.assertEqual(report["gate_b"]["domains_at_or_above_threshold"], ["small"])
        self.assertTrue(report["gate_b"]["proceed_to_causal_training"])

    def test_gate_stops_when_global_and_all_domains_are_below_threshold(self):
        report = aggregate_summaries(
            [
                summary("a", 1000, 5, [(1, 1000, 5), (2, 1000, 5)]),
                summary("b", 2000, 10, [(1, 2000, 10), (2, 2000, 10)]),
            ]
        )
        self.assertFalse(report["gate_b"]["proceed_to_causal_training"])
        self.assertEqual(
            report["gate_b"]["decision"], "停止完整重训练并报告负结果"
        )

    def test_domains_must_share_the_same_seeds(self):
        with self.assertRaisesRegex(ValueError, "相同的随机种子"):
            aggregate_summaries(
                [
                    summary("a", 10, 1, [(1, 10, 1)]),
                    summary("b", 10, 1, [(2, 10, 1)]),
                ]
            )


if __name__ == "__main__":
    unittest.main()
