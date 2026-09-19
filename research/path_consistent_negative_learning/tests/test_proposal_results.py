import json
from pathlib import Path
import tempfile
import unittest

from research.path_consistent_negative_learning.figures.gen_fig_weighted import (
    generate_figure,
)
from research.path_consistent_negative_learning.proposal.generate_results_tex import (
    build_parser,
    render_results,
)


def comparison(reference, comparison_name, metric, difference):
    return {
        "reference": reference,
        "comparison": comparison_name,
        "metric": metric,
        "seed_summary": {"mean_difference": difference},
        "paired_bootstrap": {
            "n_pairs": 8,
            "mean_difference": difference,
            "ci_low": difference - 0.005,
            "ci_high": difference + 0.005,
        },
    }


def configuration(strategy, weight, reach, precision):
    return {
        "strategy": strategy,
        "disputed_negative_weight": weight,
        "mean": {
            "answer_reach_10": reach,
            "all_shortest_recall_10": 0.6,
            "selected_pr_auc": precision,
        },
        "std": {
            "answer_reach_10": 0.01,
            "all_shortest_recall_10": 0.01,
            "selected_pr_auc": 0.01,
        },
    }


def base_reports():
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
    base_strategy = configuration("unused", None, 0.5, 0.7)
    training = {
        "strategies": {
            name: base_strategy
            for name in (
                "strategy1_negative",
                "strategy2_ignore",
                "random_drop",
                "strategy3_positive",
            )
        },
        "comparisons": [
            comparison(
                "strategy1_negative", "strategy2_ignore", "answer_reach_10", 0.02
            ),
            comparison("random_drop", "strategy2_ignore", "answer_reach_10", 0.01),
            comparison(
                "strategy1_negative", "strategy2_ignore", "selected_pr_auc", -0.005
            ),
        ],
        "gate_c": {"decision": "不扩展完整训练", "proceed_to_gate_d": False},
    }
    return audit, training


def selection_report():
    versus_baseline = comparison(
        "strategy1", "strategy2_weight_0p25", "answer_reach_10", 0.04
    )
    versus_random = comparison(
        "random_weight_0p25",
        "strategy2_weight_0p25",
        "answer_reach_10",
        0.03,
    )
    precision = comparison(
        "strategy1", "strategy2_weight_0p25", "selected_pr_auc", -0.006
    )
    return {
        "maximum_selection_precision_drop": 0.008,
        "conservative_tie_margin": 0.002,
        "configurations": {
            "strategy2_weight_0p25": configuration(
                "strategy2_weighted", 0.25, 0.64, 0.694
            ),
            "random_weight_0p25": configuration(
                "random_weighted", 0.25, 0.61, 0.699
            ),
        },
        "evaluations": [
            {
                "weight": 0.25,
                "eligible": True,
                "selection_score": 0.025,
                "answer_reach_vs_strategy1": versus_baseline,
                "answer_reach_vs_random_weighted": versus_random,
                "selected_pr_auc_vs_strategy1": precision,
            }
        ],
        "endpoint_checks_passed": True,
        "selected_weight": 0.25,
        "proceed_to_confirmation": True,
        "decision": "进入独立确认",
    }


def confirmation_report():
    return {
        "domain": "award",
        "locked_weight": 0.25,
        "maximum_confirmation_precision_drop": 0.01,
        "configurations": {
            "strategy1": configuration("strategy1_negative", None, 0.60, 0.70),
            "strategy2_weight_0p25": configuration(
                "strategy2_weighted", 0.25, 0.65, 0.695
            ),
            "random_weight_0p25": configuration(
                "random_weighted", 0.25, 0.61, 0.698
            ),
        },
        "comparisons": {
            "answer_reach_vs_strategy1": comparison(
                "strategy1", "strategy2_weight_0p25", "answer_reach_10", 0.05
            ),
            "answer_reach_vs_random_weighted": comparison(
                "random_weight_0p25",
                "strategy2_weight_0p25",
                "answer_reach_10",
                0.04,
            ),
            "selected_pr_auc_noninferiority_vs_strategy1": comparison(
                "strategy1", "strategy2_weight_0p25", "selected_pr_auc", -0.005
            ),
        },
        "gate": {"proceed_to_gate_d": True, "decision": "继续门槛D"},
    }


def gate_d_report():
    domain_comparisons = {
        "answer_reach_vs_strategy1": comparison(
            "strategy1", "strategy2_weight_0p25", "answer_reach_10", 0.05
        ),
        "answer_reach_vs_random_weighted": comparison(
            "random_weight_0p25",
            "strategy2_weight_0p25",
            "answer_reach_10",
            0.04,
        ),
        "selected_pr_auc_vs_strategy1": comparison(
            "strategy1", "strategy2_weight_0p25", "selected_pr_auc", -0.005
        ),
    }
    macro = {
        key: value for key, value in domain_comparisons.items()
    }
    return {
        "locked_weight": 0.25,
        "domains": {
            "award": {
                "configurations": {
                    "strategy2_weight_0p25": configuration(
                        "strategy2_weighted", 0.25, 0.65, 0.695
                    )
                },
                "comparisons": domain_comparisons,
            }
        },
        "macro_comparisons": macro,
        "macro_comparisons_excluding_confirmation_domain": macro,
        "direction_summary": {
            "positive_answer_reach_domains_vs_strategy1": 1,
            "domain_count": 1,
        },
    }


class ProposalResultGenerationTests(unittest.TestCase):
    def test_keeps_base_macros_and_adds_weighted_results(self):
        audit, training = base_reports()
        content = render_results(
            audit,
            training,
            selection_report(),
            confirmation_report(),
        )
        self.assertIn(r"\newcommand{\SampledRate}{10.00\%}", content)
        self.assertIn(r"\newcommand{\WeightedSelectedWeight}{0.25}", content)
        self.assertIn(r"\weightedconfirmationpassedtrue", content)
        self.assertIn("0.25 & 64.00 $\\pm$ 1.00", content)
        self.assertIn(r"\gatedavailablefalse", content)

    def test_optional_gate_d_adds_domain_and_macro_rows(self):
        audit, training = base_reports()
        content = render_results(
            audit,
            training,
            selection_report(),
            confirmation_report(),
            gate_d_report(),
        )
        self.assertIn(r"\gatedavailabletrue", content)
        self.assertIn(r"\newcommand{\GateDDomainCount}{1}", content)
        self.assertIn(
            r"\newcommand{\GateDVsStrategyOne}{+5.00 [+4.50, +5.50]}",
            content,
        )
        self.assertIn(
            r"\newcommand{\GateDPrecisionVsStrategyOne}{-0.50 [-1.00, +0.00]}",
            content,
        )
        self.assertIn("奖项 & 65.00", content)
        self.assertIn("答案可达率：相对策略1", content)

    def test_rejects_mismatched_locked_weight(self):
        audit, training = base_reports()
        confirmation = confirmation_report()
        confirmation["locked_weight"] = 0.5
        with self.assertRaisesRegex(ValueError, "锁定权重不一致"):
            render_results(audit, training, selection_report(), confirmation)

    def test_stopped_selection_needs_no_confirmation_or_gate_d(self):
        audit, training = base_reports()
        selection = selection_report()
        selection["selected_weight"] = None
        selection["proceed_to_confirmation"] = False
        selection["decision"] = "停止加权改进"
        selection["evaluations"][0]["eligible"] = False
        content = render_results(audit, training, selection)
        self.assertIn(r"\newcommand{\WeightedSelectedWeight}{--}", content)
        self.assertIn(r"\weightedselectionpassedfalse", content)
        self.assertIn(r"\weightedconfirmationavailablefalse", content)
        self.assertIn(r"\newcommand{\WeightedConfirmationDecision}{未运行}", content)
        self.assertIn(r"\gatedavailablefalse", content)

    def test_confirmation_cli_argument_is_optional(self):
        args = build_parser().parse_args(
            [
                "--audit",
                "audit.json",
                "--training",
                "training.json",
                "--selection",
                "selection.json",
            ]
        )
        self.assertIsNone(args.confirmation)

    def test_gate_d_without_confirmation_is_rejected(self):
        audit, training = base_reports()
        with self.assertRaisesRegex(ValueError, "必须同时提供独立确认"):
            render_results(
                audit,
                training,
                selection_report(),
                gate_d_report=gate_d_report(),
            )

    def test_weighted_figure_exports_pdf_and_png(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            summary_path = output_dir / "selection_summary.json"
            summary_path.write_text(
                json.dumps(selection_report(), ensure_ascii=False), encoding="utf-8"
            )
            pdf_path, png_path = generate_figure(summary_path, output_dir)
            self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF"))
            self.assertTrue(png_path.read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
