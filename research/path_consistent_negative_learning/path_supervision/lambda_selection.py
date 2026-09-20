"""仅根据独立调参划分自动选择数据集级 lambda。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping
import json


DEFAULT_LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def _metric_value(result: Mapping[str, Any], metric: str) -> float:
    if metric == "answer_reach_10":
        return float(result["metrics"]["answer_reach_at"]["10"])
    if metric.startswith("selected_"):
        key = metric.removeprefix("selected_")
        aliases = {"pr_auc": "question_pr_auc"}
        return float(result["metrics"]["selected_path"][aliases.get(key, key)])
    if metric.startswith("all_shortest_"):
        key = metric.removeprefix("all_shortest_")
        aliases = {"pr_auc": "question_pr_auc"}
        return float(
            result["metrics"]["all_shortest_paths"][aliases.get(key, key)]
        )
    raise ValueError(f"不支持的 lambda 选择指标：{metric}")


def load_selection_results(paths: Iterable[Path]) -> list[dict[str, Any]]:
    results = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("evaluation_split") != "selection":
            raise ValueError(f"lambda 选择只允许读取 selection 结果：{path}")
        if result.get("strategy") != "strategy2_weighted":
            raise ValueError(f"lambda sweep 只接受正式加权臂：{path}")
        if result.get("disputed_negative_weight") is None:
            raise ValueError(f"结果缺少 disputed_negative_weight：{path}")
        results.append(result)
    if not results:
        raise ValueError("没有可用于 lambda 选择的结果")
    return results


def select_lambda(
    results: Iterable[Mapping[str, Any]],
    *,
    selection_metric: str,
    lambda_grid: Iterable[float] = DEFAULT_LAMBDA_GRID,
) -> dict[str, Any]:
    """最大化跨种子平均验证指标；精确并列时选择更大的 lambda。"""

    records = list(results)
    if not records:
        raise ValueError("没有可用于 lambda 选择的结果")
    if any(record.get("evaluation_split") != "selection" for record in records):
        raise ValueError("检测到非 selection 结果；拒绝潜在 test leakage")

    datasets = {str(record["domain"]) for record in records}
    if len(datasets) != 1:
        raise ValueError("一次只能为一个 dataset 选择 lambda")
    grid = tuple(float(value) for value in lambda_grid)
    if len(grid) != len(set(grid)) or tuple(sorted(grid)) != grid:
        raise ValueError("lambda grid 必须严格递增且不得重复")
    if grid != DEFAULT_LAMBDA_GRID:
        raise ValueError(f"正式 lambda grid 必须是 {list(DEFAULT_LAMBDA_GRID)}")

    grouped: dict[float, dict[int, float]] = defaultdict(dict)
    for record in records:
        lambda_value = float(record["disputed_negative_weight"])
        if lambda_value not in grid:
            raise ValueError(f"结果包含协议外 lambda={lambda_value}")
        seed = int(record["seed"])
        if seed in grouped[lambda_value]:
            raise ValueError(f"lambda={lambda_value} 的 seed={seed} 重复")
        grouped[lambda_value][seed] = _metric_value(record, selection_metric)

    missing = [value for value in grid if value not in grouped]
    if missing:
        raise ValueError(f"lambda sweep 不完整，缺少：{missing}")
    seed_sets = {tuple(sorted(per_seed)) for per_seed in grouped.values()}
    if len(seed_sets) != 1:
        raise ValueError("各 lambda 必须使用完全相同的随机种子")

    scores: dict[str, dict[str, Any]] = {}
    mean_scores: dict[float, float] = {}
    for lambda_value in grid:
        per_seed = grouped[lambda_value]
        values = [per_seed[seed] for seed in sorted(per_seed)]
        mean_scores[lambda_value] = fmean(values)
        scores[str(lambda_value)] = {
            "mean": mean_scores[lambda_value],
            "std": stdev(values) if len(values) > 1 else 0.0,
            "per_seed": {str(seed): per_seed[seed] for seed in sorted(per_seed)},
        }

    best_mean = max(mean_scores.values())
    tied = [
        lambda_value
        for lambda_value, score in mean_scores.items()
        if abs(score - best_mean) <= 1e-12
    ]
    best_lambda = max(tied)
    return {
        "schema_version": 1,
        "dataset": next(iter(datasets)),
        "backbone": "structured_retriever",
        "selection_split": "selection",
        "selection_metric": selection_metric,
        "selection_direction": "maximize",
        "tie_rule": "absolute_difference_le_1e-12_then_choose_larger_lambda",
        "lambda_grid": list(grid),
        "seeds": list(next(iter(seed_sets))),
        "best_lambda": best_lambda,
        "validation_scores": scores,
        "test_metrics_accessed": False,
    }
