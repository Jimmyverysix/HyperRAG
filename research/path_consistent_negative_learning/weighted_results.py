"""Shared loading and paired statistics for weighted strategy-2 experiments."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

from .aggregate_training import _aggregate_metric_values
from .statistics import paired_bootstrap_mean_difference, summarize_paired_seeds


QUERY_METRICS = (
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


def configuration_spec(configuration: str) -> tuple[str, float | None]:
    if configuration == "strategy1":
        return "strategy1_negative", None
    if configuration == "strategy2_original":
        return "strategy2_ignore", None
    for prefix, strategy in (
        ("strategy2_weight_", "strategy2_weighted"),
        ("random_weight_", "random_weighted"),
    ):
        if configuration.startswith(prefix):
            token = configuration[len(prefix):]
            try:
                weight = float(token.replace("p", "."))
            except ValueError as exc:
                raise ValueError(f"无法解析配置目录权重：{configuration}") from exc
            return strategy, weight
    raise ValueError(f"未知配置目录：{configuration}")


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


def load_weighted_runs(
    run_root: Path,
    *,
    expected_evaluation_split: str,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Load configuration/seed runs and reject mixed evaluation protocols."""

    runs: dict[str, dict[int, dict[str, Any]]] = {}
    protocol: tuple[str, int, str, str] | None = None
    for result_path in sorted(run_root.glob("*/seed_*/result.json")):
        result = _read_json(result_path)
        try:
            directory_seed = int(result_path.parent.name.removeprefix("seed_"))
        except ValueError as exc:
            raise ValueError(f"无法解析随机种子目录：{result_path.parent}") from exc
        result_seed = int(result["seed"])
        if directory_seed != result_seed:
            raise ValueError(f"目录与结果随机种子不一致：{result_path}")
        if int(result["sampler_seed"]) != result_seed:
            raise ValueError(f"采样种子与训练种子不一致：{result_path}")
        evaluation_split = str(result.get("evaluation_split"))
        if evaluation_split != expected_evaluation_split:
            raise ValueError(
                f"{result_path} 使用 {evaluation_split!r}，"
                f"预期 {expected_evaluation_split!r}"
            )
        current_protocol = (
            str(result["domain"]),
            int(result["split_seed"]),
            str(result["split_scheme"]),
            json.dumps(result.get("training_config"), sort_keys=True),
        )
        if protocol is None:
            protocol = current_protocol
        elif current_protocol != protocol:
            raise ValueError("所有运行必须共享领域、划分种子和划分方案")
        configuration = result_path.parent.parent.name
        expected_strategy, expected_weight = configuration_spec(configuration)
        if result.get("strategy") != expected_strategy:
            raise ValueError(f"配置目录与实际策略不一致：{result_path}")
        actual_weight = result.get("disputed_negative_weight")
        if actual_weight != expected_weight:
            raise ValueError(
                f"配置目录与实际权重不一致：{result_path}: "
                f"{actual_weight!r} != {expected_weight!r}"
            )
        seed = result_seed
        configuration_runs = runs.setdefault(configuration, {})
        if seed in configuration_runs:
            raise ValueError(f"重复运行：{configuration}, seed={seed}")
        configuration_runs[seed] = {
            "result": result,
            "query_rows": _read_jsonl(result_path.with_name("query_metrics.jsonl")),
            "path": str(result_path),
        }
    if not runs:
        raise ValueError(f"没有找到训练结果：{run_root}")
    expected_seeds = set(next(iter(runs.values())))
    for configuration, seed_runs in runs.items():
        if set(seed_runs) != expected_seeds:
            raise ValueError(f"配置 {configuration} 的随机种子集合不一致")
    return runs


def summarize_configurations(
    runs: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    summaries = {}
    for configuration, seed_runs in runs.items():
        per_seed = {
            str(seed): _aggregate_metric_values(run["result"])
            for seed, run in seed_runs.items()
        }
        metric_names = tuple(next(iter(per_seed.values())))
        summaries[configuration] = {
            "strategy": next(iter(seed_runs.values()))["result"]["strategy"],
            "disputed_negative_weight": next(iter(seed_runs.values()))["result"].get(
                "disputed_negative_weight"
            ),
            "per_seed": per_seed,
            "mean": {
                metric: mean(row[metric] for row in per_seed.values())
                for metric in metric_names
            },
            "std": {
                metric: stdev(row[metric] for row in per_seed.values())
                if len(per_seed) > 1
                else 0.0
                for metric in metric_names
            },
        }
    return summaries


def _query_values(
    runs: Mapping[str, Mapping[int, Mapping[str, Any]]],
    configuration: str,
    metric: str,
) -> dict[int, dict[str, float]]:
    values_by_seed = {}
    for seed, run in runs[configuration].items():
        values = {}
        for row in run["query_rows"]:
            key = str(row["query_key"])
            if key in values:
                raise ValueError(f"重复问题：{configuration}, seed={seed}, {key}")
            values[key] = float(row[metric])
        values_by_seed[int(seed)] = values
    return values_by_seed


def _mean_queries_across_seeds(
    values: Mapping[int, Mapping[str, float]],
) -> dict[str, float]:
    seed_items = tuple(sorted(values.items()))
    expected_queries = set(seed_items[0][1])
    for seed, query_values in seed_items[1:]:
        if set(query_values) != expected_queries:
            raise ValueError(f"随机种子 {seed} 的问题集合不一致")
    return {
        query: mean(query_values[query] for _, query_values in seed_items)
        for query in seed_items[0][1]
    }


def compare_configurations(
    runs: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    reference: str,
    comparison: str,
    metric: str,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 20260919,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    reference_values = _query_values(runs, reference, metric)
    comparison_values = _query_values(runs, comparison, metric)
    seed_summary = summarize_paired_seeds(reference_values, comparison_values)
    bootstrap = paired_bootstrap_mean_difference(
        _mean_queries_across_seeds(reference_values),
        _mean_queries_across_seeds(comparison_values),
        confidence_level=confidence_level,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    seed_payload = asdict(seed_summary)
    seed_payload["per_seed"] = {
        str(seed): asdict(value) for seed, value in seed_summary.per_seed.items()
    }
    return {
        "reference": reference,
        "comparison": comparison,
        "metric": metric,
        "bootstrap_unit": "query_after_equal_seed_average",
        "seed_summary": seed_payload,
        "paired_bootstrap": asdict(bootstrap),
    }


def compare_query_rows_exact(
    runs: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    reference: str,
    comparison: str,
    metrics: tuple[str, ...],
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Check endpoint equivalence without allowing query differences to cancel."""

    if set(runs[reference]) != set(runs[comparison]):
        raise ValueError("端点参照与比较的随机种子集合不一致")
    details: dict[str, Any] = {}
    passed = True
    for seed in sorted(runs[reference]):
        reference_rows = {
            str(row["query_key"]): row for row in runs[reference][seed]["query_rows"]
        }
        comparison_rows = {
            str(row["query_key"]): row for row in runs[comparison][seed]["query_rows"]
        }
        if set(reference_rows) != set(comparison_rows):
            raise ValueError(f"端点 seed={seed} 的问题集合不一致")
        seed_details = {}
        for metric in metrics:
            differences = [
                abs(float(comparison_rows[key][metric]) - float(reference_rows[key][metric]))
                for key in reference_rows
            ]
            maximum = max(differences, default=0.0)
            changed = sum(value > tolerance for value in differences)
            seed_details[metric] = {
                "maximum_absolute_difference": maximum,
                "changed_query_count": changed,
            }
            passed &= changed == 0
        details[str(seed)] = seed_details
    return {
        "passed": passed,
        "tolerance": tolerance,
        "per_seed": details,
    }


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
