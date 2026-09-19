import unittest

from research.path_consistent_negative_learning.paper.generate_results_tex import (
    render_results,
)


class PaperResultGenerationTests(unittest.TestCase):
    def test_rendered_values_and_gate_are_data_driven(self):
        audit = {
            "overall": {
                "domain_count": 1,
                "query_count": 10,
                "candidate_count": 100,
                "disputed_candidate_count": 2,
                "candidate_rate": 0.02,
                "sampled_count": 50,
                "disputed_sampled_count": 5,
                "sampled_rate": 0.1,
            },
            "gate_b": {"threshold": 0.01, "decision": "继续门槛C"},
            "domains": [
                {
                    "domain": "edu",
                    "query_count": 10,
                    "candidate_rate": 0.02,
                    "sampled_rate": 0.1,
                    "sampled_seed_mean": 0.1,
                    "sampled_seed_std": 0.01,
                }
            ],
        }
        strategy = {
            "mean": {
                "answer_reach_10": 0.5,
                "all_shortest_recall_10": 0.6,
                "selected_pr_auc": 0.7,
            },
            "std": {
                "answer_reach_10": 0.01,
                "all_shortest_recall_10": 0.02,
                "selected_pr_auc": 0.03,
            },
        }

        def comparison(reference, metric, difference):
            return {
                "reference": reference,
                "comparison": "strategy2_ignore",
                "metric": metric,
                "paired_bootstrap": {
                    "n_pairs": 20,
                    "mean_difference": difference,
                    "ci_low": difference - 0.01,
                    "ci_high": difference + 0.01,
                },
                "seed_summary": {"mean_difference": difference},
            }

        training = {
            "strategies": {
                name: strategy
                for name in (
                    "strategy1_negative",
                    "strategy2_ignore",
                    "random_drop",
                    "strategy3_positive",
                )
            },
            "comparisons": [
                comparison("strategy1_negative", "answer_reach_10", 0.02),
                comparison("random_drop", "answer_reach_10", 0.01),
                comparison("strategy1_negative", "selected_pr_auc", -0.005),
            ],
            "gate_c": {
                "decision": "不扩展完整训练",
                "proceed_to_gate_d": False,
            },
        }

        content = render_results(audit, training)

        self.assertIn(r"\newcommand{\SampledRate}{10.00\%}", content)
        self.assertIn("教育 & 10", content)
        self.assertIn(r"\gatecpassedfalse", content)


if __name__ == "__main__":
    unittest.main()
