"""Aggregate multi-seed Gate C runs and make the preregistered Gate D decision."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from .statistics import (
    paired_bootstrap_mean_difference,
    summarize_paired_seeds,
)
from .strategies import STRATEGIES


PRIMARY_METRIC = "answer_reach_10"
PRECISION_GUARD_METRIC = "selected_pr_auc"
MAX_PRECISION_DROP = 0.01
COMPARISONS = (
    ("strategy1_negative", "strategy2_ignore"),
    ("strategy3_positive", "strategy2_ignore"),
    ("random_drop", "strategy2_ignore"),
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _aggregate_metric_values(result: Mapping[str, Any]) -> dict[str, float]:
    metrics = result["metrics"]
    selected = metrics["selected_path"]
    all_shortest = metrics["all_shortest_paths"]
    return {
        "selected_mrr": float(selected["mrr"]),
        "selected_hits_10": float(selected["hits_at"]["10"]),
        "selected_recall_10": float(selected["recall_at"]["10"]),
        "selected_pr_auc": float(selected["question_pr_auc"]),
        "all_shortest_mrr": float(all_shortest["mrr"]),
        "all_shortest_hits_10": float(all_shortest["hits_at"]["10"]),
        "all_shortest_recall_10": float(all_shortest["recall_at"]["10"]),
        "all_shortest_pr_auc": float(all_shortest["question_pr_auc"]),
        "answer_reach_10": float(metrics["answer_reach_at"]["10"]),
        "training_seconds": float(result["training_seconds"]),
        "peak_cuda_memory_bytes": float(result["peak_cuda_memory_bytes"]),
    }


def load_runs(run_root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    runs: dict[str, dict[int, dict[str, Any]]] = {strategy: {} for strategy in STRATEGIES}
    for result_path in sorted(run_root.glob("*/seed_*/result.json")):
        result = _read_json(result_path)
        strategy = str(result["strategy"])
        seed = int(result["seed"])
        if strategy not in runs:
            raise ValueError(f"未知策略：{strategy}")
        if seed in runs[strategy]:
            raise ValueError(f"重复运行：{strategy}, seed={seed}")
        runs[strategy][seed] = {
            "result": result,
            "query_rows": _read_jsonl(result_path.with_name("query_metrics.jsonl")),
            "path": str(result_path),
        }
    expected_seeds = set(runs[STRATEGIES[0]])
    if not expected_seeds:
        raise ValueError(f"没有找到训练结果：{run_root}")
    for strategy in STRATEGIES:
        if set(runs[strategy]) != expected_seeds:
            raise ValueError("四个策略必须具有完全相同的随机种子")
    return runs


def _query_values(
    runs: Mapping[str, Mapping[int, Mapping[str, Any]]],
    strategy: str,
    metric: str,
) -> dict[int, dict[str, float]]:
    output = {}
    for seed, run in runs[strategy].items():
        values = {}
        for row in run["query_rows"]:
            key = str(row["query_key"])
            if key in values:
                raise ValueError(f"重复问题：{strategy}, seed={seed}, {key}")
            values[key] = float(row[metric])
        output[int(seed)] = values
    return output


def _comparison(
    runs: Mapping[str, Mapping[int, Mapping[str, Any]]],
    reference: str,
    comparison: str,
    metric: str,
) -> dict[str, Any]:
    reference_values = _query_values(runs, reference, metric)
    comparison_values = _query_values(runs, comparison, metric)
    seed_summary = summarize_paired_seeds(reference_values, comparison_values)
    pooled_reference = {
        (seed, query): value
        for seed, values in reference_values.items()
        for query, value in values.items()
    }
    pooled_comparison = {
        (seed, query): value
        for seed, values in comparison_values.items()
        for query, value in values.items()
    }
    bootstrap = paired_bootstrap_mean_difference(
        pooled_reference,
        pooled_comparison,
        n_resamples=10_000,
        seed=20260919,
    )
    seed_payload = asdict(seed_summary)
    seed_payload["per_seed"] = {
        str(seed): asdict(value) for seed, value in seed_summary.per_seed.items()
    }
    return {
        "reference": reference,
        "comparison": comparison,
        "metric": metric,
        "seed_summary": seed_payload,
        "paired_bootstrap": asdict(bootstrap),
    }


def aggregate_runs(run_root: Path) -> dict[str, Any]:
    runs = load_runs(run_root)
    strategies: dict[str, Any] = {}
    metric_names: tuple[str, ...] | None = None
    for strategy, seed_runs in runs.items():
        per_seed = {
            seed: _aggregate_metric_values(run["result"])
            for seed, run in seed_runs.items()
        }
        metric_names = tuple(next(iter(per_seed.values())))
        strategies[strategy] = {
            "per_seed": {str(seed): values for seed, values in per_seed.items()},
            "mean": {
                metric: mean(values[metric] for values in per_seed.values())
                for metric in metric_names
            },
            "std": {
                metric: stdev(values[metric] for values in per_seed.values())
                if len(per_seed) > 1
                else 0.0
                for metric in metric_names
            },
        }

    query_metric_names = (
        "selected_mrr",
        "selected_pr_auc",
        "selected_hits_10",
        "selected_recall_10",
        "all_shortest_mrr",
        "all_shortest_pr_auc",
        "all_shortest_hits_10",
        "all_shortest_recall_10",
        "answer_reach_10",
    )
    comparisons = [
        _comparison(runs, reference, comparison, metric)
        for reference, comparison in COMPARISONS
        for metric in query_metric_names
    ]

    def comparison_result(reference: str, metric: str) -> Mapping[str, Any]:
        return next(
            row
            for row in comparisons
            if row["reference"] == reference
            and row["comparison"] == "strategy2_ignore"
            and row["metric"] == metric
        )

    baseline_primary = comparison_result("strategy1_negative", PRIMARY_METRIC)
    random_primary = comparison_result("random_drop", PRIMARY_METRIC)
    precision_guard = comparison_result(
        "strategy1_negative", PRECISION_GUARD_METRIC
    )
    primary_vs_baseline = baseline_primary["paired_bootstrap"]
    primary_vs_random = random_primary["paired_bootstrap"]
    precision_difference = precision_guard["seed_summary"]["mean_difference"]
    proceed = (
        primary_vs_baseline["ci_low"] > 0.0
        and primary_vs_random["ci_low"] > 0.0
        and precision_difference >= -MAX_PRECISION_DROP
    )
    return {
        "schema_version": 1,
        "experiment": "structured_proxy_gate_c",
        "strategies": strategies,
        "comparisons": comparisons,
        "gate_c": {
            "primary_metric": PRIMARY_METRIC,
            "precision_guard_metric": PRECISION_GUARD_METRIC,
            "maximum_allowed_precision_drop": MAX_PRECISION_DROP,
            "strategy2_vs_strategy1_primary_ci_low": primary_vs_baseline["ci_low"],
            "strategy2_vs_random_drop_primary_ci_low": primary_vs_random["ci_low"],
            "strategy2_vs_strategy1_precision_difference": precision_difference,
            "proceed_to_gate_d": proceed,
            "decision": "继续门槛D" if proceed else "不扩展完整训练",
        },
    }


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    rows = []
    for strategy, values in report["strategies"].items():
        row = {"strategy": strategy}
        for metric, value in values["mean"].items():
            row[f"{metric}_mean"] = value
            row[f"{metric}_std"] = values["std"][metric]
        rows.append(row)
    with (output_dir / "training_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = aggregate_runs(args.run_root)
    write_report(report, args.output_dir)
    print(json.dumps(report["gate_c"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
