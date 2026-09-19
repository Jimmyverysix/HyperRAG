import json
from pathlib import Path
import tempfile
import unittest

from research.path_consistent_negative_learning.aggregate_gate_d import (
    aggregate_gate_d,
)
from research.path_consistent_negative_learning.confirm_weight import confirm_weight
from research.path_consistent_negative_learning.select_weight import (
    choose_weight,
    select_weight,
)
from research.path_consistent_negative_learning.weighted_results import (
    compare_query_rows_exact,
    load_weighted_runs,
)


def write_run(
    root: Path,
    configuration: str,
    seed: int,
    *,
    strategy: str,
    weight: float | None,
    evaluation_split: str,
    answer_reach: float,
    selected_pr_auc: float,
    domain: str = "toy",
    experiment_name: str | None = None,
) -> None:
    run_dir = root / configuration / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    result = {
        "domain": domain,
        "strategy": strategy,
        "seed": seed,
        "sampler_seed": seed,
        "split_seed": 99,
        "split_scheme": (
            "60/10/15/15" if evaluation_split == "selection" else "70/15/15"
        ),
        "evaluation_split": evaluation_split,
        "disputed_negative_weight": weight,
        "experiment": experiment_name or (
            f"weighted_strategy2_{'selection' if evaluation_split == 'selection' else 'confirmation'}"
        ),
        "training_config": {
            "epochs": 50,
            "patience": 8,
            "batch_size": 4096,
            "learning_rate": 0.001,
        },
        "training_seconds": 1.0,
        "peak_cuda_memory_bytes": 10.0,
        "metrics": {
            "selected_path": {
                "mrr": 0.8,
                "hits_at": {"10": 1.0},
                "recall_at": {"10": 0.7},
                "question_pr_auc": selected_pr_auc,
            },
            "all_shortest_paths": {
                "mrr": 0.8,
                "hits_at": {"10": 1.0},
                "recall_at": {"10": 0.7},
                "question_pr_auc": 0.8,
            },
            "answer_reach_at": {"10": answer_reach},
        },
    }
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    rows = []
    for query_index in range(8):
        rows.append(
            {
                "query_key": f"q{query_index}",
                "selected_mrr": 0.8,
                "selected_pr_auc": selected_pr_auc,
                "selected_hits_10": 1.0,
                "selected_recall_10": 0.7,
                "all_shortest_mrr": 0.8,
                "all_shortest_pr_auc": 0.8,
                "all_shortest_hits_10": 1.0,
                "all_shortest_recall_10": 0.7,
                "answer_reach_10": answer_reach,
            }
        )
    (run_dir / "query_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_protocol(path: Path) -> None:
    training = {
        "epochs": 50,
        "patience": 8,
        "batch_size": 4096,
        "learning_rate": 0.001,
    }
    payload = {
        "selection": {
            "domain": "toy",
            "sampler_and_training_seeds": [42, 43],
            "split_seed": 99,
            "split_scheme": "60/10/15/15",
            "reported_split": "selection",
            "candidate_weights": [0.25, 0.5],
            "diagnostic_endpoint_weights": [0.0, 1.0],
            "training": training,
        },
        "independent_confirmation": {
            "domain": "toy",
            "sampler_and_training_seeds": [142, 143],
            "split_seed": 99,
            "split_scheme": "70/15/15",
            "reported_split": "test",
            "training": training,
        },
        "gate_d": {
            "domains": ["d1", "d2"],
            "sampler_and_training_seeds": [142, 143],
            "split_seed": 99,
            "split_scheme": "70/15/15",
            "reported_split": "test",
            "training": training,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class WeightedProtocolTests(unittest.TestCase):
    def test_gate_d_reports_domain_macro_and_excludes_confirmation_domain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            write_protocol(protocol_path)
            for domain in ("d1", "d2"):
                run_root = root / domain / "runs"
                for seed in (142, 143):
                    for configuration, strategy, weight, reach, precision in (
                        ("strategy1", "strategy1_negative", None, 0.50, 0.60),
                        (
                            "strategy2_weight_0p25",
                            "strategy2_weighted",
                            0.25,
                            0.60,
                            0.595,
                        ),
                        (
                            "random_weight_0p25",
                            "random_weighted",
                            0.25,
                            0.52,
                            0.598,
                        ),
                    ):
                        write_run(
                            run_root,
                            configuration,
                            seed,
                            strategy=strategy,
                            weight=weight,
                            evaluation_split="test",
                            answer_reach=reach,
                            selected_pr_auc=precision,
                            domain=domain,
                            experiment_name="weighted_strategy2_gate_d",
                        )
            report = aggregate_gate_d(
                root,
                locked_weight=0.25,
                protocol_path=protocol_path,
            )
        macro = report["macro_comparisons"]["answer_reach_vs_strategy1"]
        self.assertAlmostEqual(
            macro["paired_bootstrap"]["mean_difference"],
            0.10,
        )

    def test_endpoint_check_rejects_query_differences_that_cancel_in_mean(self):
        runs = {
            "reference": {
                1: {
                    "query_rows": [
                        {"query_key": "a", "metric": 0.4},
                        {"query_key": "b", "metric": 0.6},
                    ]
                }
            },
            "comparison": {
                1: {
                    "query_rows": [
                        {"query_key": "a", "metric": 0.5},
                        {"query_key": "b", "metric": 0.5},
                    ]
                }
            },
        }
        result = compare_query_rows_exact(
            runs,
            reference="reference",
            comparison="comparison",
            metrics=("metric",),
        )
        self.assertFalse(result["passed"])

    def test_tie_rule_is_relative_to_global_best_not_previous_candidate(self):
        selected = choose_weight(
            (
                {"weight": 0.75, "selection_score": 0.0, "eligible": True},
                {"weight": 0.50, "selection_score": 0.0015, "eligible": True},
                {"weight": 0.25, "selection_score": 0.0030, "eligible": True},
            )
        )
        self.assertEqual(selected["weight"], 0.50)

    def test_selection_uses_only_selection_split_and_locks_best_weight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            write_protocol(protocol_path)
            for seed in (42, 43):
                write_run(
                    root,
                    "strategy1",
                    seed,
                    strategy="strategy1_negative",
                    weight=None,
                    evaluation_split="selection",
                    answer_reach=0.5,
                    selected_pr_auc=0.6,
                )
                write_run(
                    root,
                    "strategy2_original",
                    seed,
                    strategy="strategy2_ignore",
                    weight=None,
                    evaluation_split="selection",
                    answer_reach=0.72,
                    selected_pr_auc=0.58,
                )
                write_run(
                    root,
                    "strategy2_weight_0",
                    seed,
                    strategy="strategy2_weighted",
                    weight=0.0,
                    evaluation_split="selection",
                    answer_reach=0.72,
                    selected_pr_auc=0.58,
                )
                write_run(
                    root,
                    "strategy2_weight_1",
                    seed,
                    strategy="strategy2_weighted",
                    weight=1.0,
                    evaluation_split="selection",
                    answer_reach=0.5,
                    selected_pr_auc=0.6,
                )
                for weight, reach, random_reach, precision in (
                    (0.25, 0.70, 0.51, 0.595),
                    (0.50, 0.65, 0.52, 0.593),
                ):
                    token = str(weight).replace(".", "p").rstrip("0")
                    write_run(
                        root,
                        f"strategy2_weight_{token}",
                        seed,
                        strategy="strategy2_weighted",
                        weight=weight,
                        evaluation_split="selection",
                        answer_reach=reach,
                        selected_pr_auc=precision,
                    )
                    write_run(
                        root,
                        f"random_weight_{token}",
                        seed,
                        strategy="random_weighted",
                        weight=weight,
                        evaluation_split="selection",
                        answer_reach=random_reach,
                        selected_pr_auc=0.598,
                    )
            report = select_weight(
                root,
                candidate_weights=(0.25, 0.5),
                protocol_path=protocol_path,
            )
        self.assertTrue(report["proceed_to_confirmation"])
        self.assertEqual(report["selected_weight"], 0.25)

    def test_confirmation_applies_coverage_and_precision_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            write_protocol(protocol_path)
            for seed in (142, 143):
                write_run(
                    root,
                    "strategy1",
                    seed,
                    strategy="strategy1_negative",
                    weight=None,
                    evaluation_split="test",
                    answer_reach=0.50,
                    selected_pr_auc=0.60,
                )
                write_run(
                    root,
                    "strategy2_weight_0p25",
                    seed,
                    strategy="strategy2_weighted",
                    weight=0.25,
                    evaluation_split="test",
                    answer_reach=0.70,
                    selected_pr_auc=0.595,
                )
                write_run(
                    root,
                    "random_weight_0p25",
                    seed,
                    strategy="random_weighted",
                    weight=0.25,
                    evaluation_split="test",
                    answer_reach=0.52,
                    selected_pr_auc=0.598,
                )
            report = confirm_weight(
                root,
                locked_weight=0.25,
                protocol_path=protocol_path,
            )
        self.assertTrue(report["gate"]["proceed_to_gate_d"])

    def test_loader_rejects_test_results_during_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_run(
                root,
                "strategy1",
                42,
                strategy="strategy1_negative",
                weight=None,
                evaluation_split="test",
                answer_reach=0.5,
                selected_pr_auc=0.6,
            )
            with self.assertRaisesRegex(ValueError, "预期 'selection'"):
                load_weighted_runs(root, expected_evaluation_split="selection")


if __name__ == "__main__":
    unittest.main()
