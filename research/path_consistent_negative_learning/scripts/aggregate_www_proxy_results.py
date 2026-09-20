"""从冻结后的 test raw runs 生成主表、配对统计和探索性相关分析。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr

from research.path_consistent_negative_learning.statistics import (
    paired_bootstrap_mean_difference,
)
from research.path_consistent_negative_learning.weighted_results import (
    compare_configurations,
    summarize_configurations,
)


ARMS = (
    "baseline_lambda_1",
    "mask_endpoint_lambda_0",
    "dataset_specific_lambda_star",
    "matched_random_at_lambda_star",
    "all_shortest_positive",
)
METRICS = (
    "answer_reach_10",
    "selected_mrr",
    "selected_pr_auc",
    "all_shortest_mrr",
    "all_shortest_pr_auc",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_domain_runs(
    domain_root: Path,
    *,
    domain: str,
    expected_seeds: list[int],
    split_seed: int,
    split_scheme: str,
    lambda_star: float,
) -> dict[str, dict[int, dict[str, Any]]]:
    runs: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in ARMS:
        arm_runs = {}
        for seed in expected_seeds:
            result_path = domain_root / arm / f"seed_{seed}" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            actual = (
                result["domain"],
                int(result["seed"]),
                int(result["sampler_seed"]),
                int(result["split_seed"]),
                result["split_scheme"],
                result["evaluation_split"],
            )
            expected = (domain, seed, seed, split_seed, split_scheme, "test")
            if actual != expected:
                raise ValueError(f"test 结果不符合冻结协议：{result_path}")
            arm_runs[seed] = {
                "result": result,
                "query_rows": _read_jsonl(
                    result_path.with_name("query_metrics.jsonl")
                ),
                "path": str(result_path),
            }
        runs[arm] = arm_runs

    expected_weights = {
        "baseline_lambda_1": 1.0,
        "mask_endpoint_lambda_0": 0.0,
        "dataset_specific_lambda_star": lambda_star,
        "matched_random_at_lambda_star": lambda_star,
        "all_shortest_positive": None,
    }
    for arm, weight in expected_weights.items():
        if any(
            seed_run["result"].get("disputed_negative_weight") != weight
            for seed_run in runs[arm].values()
        ):
            raise ValueError(f"{domain}/{arm} 的 lambda 与冻结值不一致")
    return runs


def _macro_comparison(
    domain_reports: dict[str, Any],
    *,
    reference: str,
    comparison: str,
    metric: str,
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    reference_values = {
        domain: report["arms"][reference]["mean"][metric]
        for domain, report in domain_reports.items()
    }
    comparison_values = {
        domain: report["arms"][comparison]["mean"][metric]
        for domain, report in domain_reports.items()
    }
    return asdict(
        paired_bootstrap_mean_difference(
            reference_values,
            comparison_values,
            n_resamples=n_resamples,
            seed=seed,
        )
    )


def aggregate(
    config_path: Path,
    run_root: Path,
    lambda_root: Path,
    audit_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rates = {
        row["domain"]: float(row["sampled_conflict_rate"])
        for row in audit["domains"]
    }
    seeds = [int(value) for value in config["randomness"]["sampler_and_training_seeds"]]
    bootstrap_seed = int(config["randomness"]["bootstrap_seed"])
    n_resamples = int(config["randomness"]["bootstrap_resamples"])
    domain_reports = {}
    table_rows = []
    comparison_rows = []

    for domain in config["datasets"]:
        selection_path = lambda_root / domain / "lambda_star.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("test_metrics_accessed") is not False:
            raise ValueError(f"lambda freeze 声明无效：{selection_path}")
        lambda_star = float(selection["best_lambda"])
        runs = _load_domain_runs(
            run_root / "test" / domain,
            domain=domain,
            expected_seeds=seeds,
            split_seed=int(config["randomness"]["split_seed"]),
            split_scheme=config["randomness"]["split_scheme"],
            lambda_star=lambda_star,
        )
        arms = summarize_configurations(runs)
        comparisons = {}
        for metric in METRICS:
            for reference, comparison in (
                ("baseline_lambda_1", "dataset_specific_lambda_star"),
                ("matched_random_at_lambda_star", "dataset_specific_lambda_star"),
                ("baseline_lambda_1", "mask_endpoint_lambda_0"),
                ("baseline_lambda_1", "all_shortest_positive"),
            ):
                key = f"{comparison}_vs_{reference}_{metric}"
                report = compare_configurations(
                    runs,
                    reference=reference,
                    comparison=comparison,
                    metric=metric,
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_resamples=n_resamples,
                )
                comparisons[key] = report
                paired = report["paired_bootstrap"]
                comparison_rows.append(
                    {
                        "dataset": domain,
                        "reference": reference,
                        "comparison": comparison,
                        "metric": metric,
                        "difference": paired["mean_difference"],
                        "ci_low": paired["ci_low"],
                        "ci_high": paired["ci_high"],
                    }
                )
        domain_reports[domain] = {
            "selected_lambda": lambda_star,
            "sampled_conflict_rate": rates[domain],
            "arms": arms,
            "comparisons": comparisons,
        }
        baseline = arms["baseline_lambda_1"]
        tuned = arms["dataset_specific_lambda_star"]
        random_control = arms["matched_random_at_lambda_star"]
        mask = arms["mask_endpoint_lambda_0"]
        all_positive = arms["all_shortest_positive"]
        table_rows.append(
            {
                "dataset": domain,
                "selected_lambda": lambda_star,
                "sampled_conflict_rate": rates[domain],
                "baseline_answer_reach_10_mean": baseline["mean"]["answer_reach_10"],
                "baseline_answer_reach_10_std": baseline["std"]["answer_reach_10"],
                "tuned_answer_reach_10_mean": tuned["mean"]["answer_reach_10"],
                "tuned_answer_reach_10_std": tuned["std"]["answer_reach_10"],
                "tuned_minus_baseline_answer_reach_10": tuned["mean"]["answer_reach_10"]
                - baseline["mean"]["answer_reach_10"],
                "random_answer_reach_10_mean": random_control["mean"]["answer_reach_10"],
                "mask_answer_reach_10_mean": mask["mean"]["answer_reach_10"],
                "all_positive_answer_reach_10_mean": all_positive["mean"]["answer_reach_10"],
                "baseline_selected_pr_auc_mean": baseline["mean"]["selected_pr_auc"],
                "tuned_selected_pr_auc_mean": tuned["mean"]["selected_pr_auc"],
                "tuned_minus_baseline_selected_pr_auc": tuned["mean"]["selected_pr_auc"]
                - baseline["mean"]["selected_pr_auc"],
            }
        )

    macro = {}
    for metric in METRICS:
        for reference, comparison in (
            ("baseline_lambda_1", "dataset_specific_lambda_star"),
            ("matched_random_at_lambda_star", "dataset_specific_lambda_star"),
            ("baseline_lambda_1", "mask_endpoint_lambda_0"),
            ("baseline_lambda_1", "all_shortest_positive"),
        ):
            key = f"{comparison}_vs_{reference}_{metric}"
            macro[key] = _macro_comparison(
                domain_reports,
                reference=reference,
                comparison=comparison,
                metric=metric,
                seed=bootstrap_seed,
                n_resamples=n_resamples,
            )

    lambdas = [domain_reports[domain]["selected_lambda"] for domain in config["datasets"]]
    conflict_rates = [rates[domain] for domain in config["datasets"]]
    correlation = spearmanr(conflict_rates, lambdas)
    correlation_report = {
        "analysis": "exploratory_not_causal",
        "n_datasets": len(lambdas),
        "spearman_rho": float(correlation.statistic),
        "p_value": float(correlation.pvalue),
    }
    report = {
        "schema_version": 1,
        "experiment": "www_revision_structured_proxy",
        "scope": config["scope"],
        "domains": domain_reports,
        "macro_comparisons": macro,
        "lambda_conflict_correlation": correlation_report,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "proxy_main_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "lambda_conflict_correlation.json").write_text(
        json.dumps(correlation_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "proxy_main_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    with (output_dir / "proxy_comparisons.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lambda-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = aggregate(
        args.config, args.run_root, args.lambda_root, args.audit, args.output_dir
    )
    print(json.dumps(report["macro_comparisons"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
