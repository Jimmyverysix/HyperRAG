"""Validation-only per-domain lambda selection for Retriever-only runs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping
import json


LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
SEEDS = (42, 43, 44, 45, 46)


def load_validation_reports(paths: Iterable[Path]) -> list[dict[str, Any]]:
    reports = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("stage") != "evaluate" or value.get("split") != "valid":
            raise ValueError(f"selection accepts only valid evaluation reports: {path}")
        if value.get("method") != "ours":
            raise ValueError(f"selection accepts only the path-aware lambda sweep: {path}")
        reports.append(value)
    if not reports:
        raise ValueError("no validation reports were provided")
    return reports


def select_domain_lambda(
    reports: Iterable[Mapping[str, Any]],
    *,
    lambda_grid: tuple[float, ...] = LAMBDA_GRID,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    values = list(reports)
    domains = {str(value["domain"]) for value in values}
    if len(domains) != 1:
        raise ValueError("one selection call must contain exactly one domain")
    grouped: dict[float, dict[int, float]] = defaultdict(dict)
    for value in values:
        lambda_ = float(value["lambda"])
        seed = int(value["seed"])
        if lambda_ not in lambda_grid or seed not in seeds:
            raise ValueError("validation report is outside the frozen grid or seeds")
        if seed in grouped[lambda_]:
            raise ValueError(f"duplicate validation result for lambda={lambda_}, seed={seed}")
        grouped[lambda_][seed] = float(value["metrics"]["answer_path_mrr"])
    expected_seeds = set(seeds)
    for lambda_ in lambda_grid:
        if set(grouped[lambda_]) != expected_seeds:
            raise ValueError(f"incomplete validation sweep for lambda={lambda_}")

    summaries = {}
    means = {}
    for lambda_ in lambda_grid:
        per_seed = grouped[lambda_]
        scores = [per_seed[seed] for seed in seeds]
        means[lambda_] = fmean(scores)
        summaries[str(lambda_)] = {
            "mean_answer_path_mrr": means[lambda_],
            "standard_deviation": stdev(scores),
            "per_seed": {str(seed): per_seed[seed] for seed in seeds},
        }
    best_mean = max(means.values())
    selected = max(
        lambda_ for lambda_, value in means.items() if abs(value - best_mean) <= 1e-12
    )
    return {
        "domain": next(iter(domains)),
        "lambda": selected,
        "selection_split": "valid",
        "selection_metric": "answer_path_mrr",
        "tie_rule": "absolute_difference_le_1e-12_then_choose_larger_lambda",
        "validation": summaries,
    }


def select_all_domains(run_root: Path, domains: Iterable[str]) -> dict[str, Any]:
    selected = {}
    for domain in domains:
        reports = load_validation_reports(
            sorted(
                (run_root / "sweep" / domain).glob(
                    "lambda_*/seed_*/valid_scores.report.json"
                )
            )
        )
        selected[domain] = select_domain_lambda(reports)
    return {
        "schema_version": 1,
        "status": "frozen_before_test",
        "lambda_grid": list(LAMBDA_GRID),
        "seeds": list(SEEDS),
        "domains": selected,
        "test_metrics_accessed": False,
    }
