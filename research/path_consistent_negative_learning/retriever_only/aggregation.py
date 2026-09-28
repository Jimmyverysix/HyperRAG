"""Aggregate formal Retriever-only prevalence, test, and sensitivity results."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping

import numpy as np

from .prepared import PreparedCandidates


METRICS = ("reciprocal_rank", "answer_reach_10", "answer_reach_5")
FORMAL_SEEDS = (42, 43, 44, 45, 46)


def aggregate_conflict_prevalence(
    run_root: Path,
    domains: Iterable[str],
    seeds: Iterable[int],
) -> dict[str, Any]:
    rows = []
    total_negative = 0
    total_conflict = 0
    for domain in domains:
        domain_negative = 0
        domain_conflict = 0
        all_query_keys: set[str] = set()
        affected_query_keys: set[str] = set()
        per_seed = []
        for seed in seeds:
            data = PreparedCandidates.load(
                run_root / "prepared" / "train" / domain / f"seed_{seed}.pt"
            )
            negative = int((~data.labels).sum())
            conflict = int(data.path_consistent_mask.sum())
            affected = 0
            for index, key in enumerate(data.query_keys):
                start = int(data.query_offsets[index])
                stop = int(data.query_offsets[index + 1])
                all_query_keys.add(key)
                if bool(data.path_consistent_mask[start:stop].any()):
                    affected += 1
                    affected_query_keys.add(key)
            domain_negative += negative
            domain_conflict += conflict
            per_seed.append(
                {
                    "seed": seed,
                    "sampled_negative_count": negative,
                    "path_consistent_negative_count": conflict,
                    "path_consistent_negative_rate": conflict / negative,
                    "affected_query_count": affected,
                    "affected_query_rate": affected / len(data.query_keys),
                }
            )
        total_negative += domain_negative
        total_conflict += domain_conflict
        rows.append(
            {
                "domain": domain,
                "sampled_negative_count": domain_negative,
                "path_consistent_negative_count": domain_conflict,
                "path_consistent_negative_rate": domain_conflict / domain_negative,
                "query_count": len(all_query_keys),
                "affected_query_count_across_seeds": len(affected_query_keys),
                "affected_query_rate_across_seeds": len(affected_query_keys)
                / len(all_query_keys),
                "per_seed": per_seed,
            }
        )
    return {
        "domains": rows,
        "overall": {
            "sampled_negative_count": total_negative,
            "path_consistent_negative_count": total_conflict,
            "path_consistent_negative_rate": total_conflict / total_negative,
            "equal_domain_affected_query_rate": fmean(
                row["affected_query_rate_across_seeds"] for row in rows
            ),
        },
    }


def _load_method_reports(
    run_root: Path,
    domain: str,
    method: str,
) -> list[dict[str, Any]]:
    paths = sorted(
        (run_root / "test" / domain / method).glob(
            "seed_*/test_scores.report.json"
        )
    )
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not reports or any(report.get("split") != "test" for report in reports):
        raise ValueError(f"missing or non-test reports for {domain}/{method}")
    if any(report.get("method") != method for report in reports):
        raise ValueError(f"method metadata mismatch for {domain}/{method}")
    if {int(report["seed"]) for report in reports} != set(FORMAL_SEEDS):
        raise ValueError(f"incomplete formal seed set for {domain}/{method}")
    return reports


def average_query_metrics(reports: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    by_query: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    reports = list(reports)
    for report in reports:
        for query in report["queries"]:
            for metric in METRICS:
                by_query[str(query["query_key"])][metric].append(float(query[metric]))
    expected = len(reports)
    output = {}
    for query_key, values in by_query.items():
        if any(len(values[metric]) != expected for metric in METRICS):
            raise ValueError(f"query seed coverage mismatch: {query_key}")
        output[query_key] = {
            metric: fmean(values[metric]) for metric in METRICS
        }
    return output


def paired_equal_domain_bootstrap(
    differences: Mapping[str, np.ndarray],
    *,
    resamples: int = 10_000,
    seed: int = 20_260_928,
    batch_size: int = 100,
) -> dict[str, float]:
    if not differences:
        raise ValueError("at least one domain difference vector is required")
    rng = np.random.default_rng(seed)
    distribution = np.zeros(resamples, dtype=np.float64)
    for values in differences.values():
        if values.ndim != 1 or not len(values):
            raise ValueError("bootstrap difference vectors must be nonempty and 1-D")
        domain_means = np.empty(resamples, dtype=np.float64)
        for start in range(0, resamples, batch_size):
            stop = min(start + batch_size, resamples)
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            domain_means[start:stop] = values[indices].mean(axis=1)
        distribution += domain_means
    distribution /= len(differences)
    return {
        "difference": float(
            fmean(float(values.mean()) for values in differences.values())
        ),
        "ci_low": float(np.quantile(distribution, 0.025)),
        "ci_high": float(np.quantile(distribution, 0.975)),
        "resamples": resamples,
        "seed": seed,
    }


def aggregate_main_test(
    run_root: Path,
    domains: Iterable[str],
) -> dict[str, Any]:
    methods = ("baseline", "matched_random", "ours")
    query_values: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    domain_rows = []
    for domain in domains:
        query_values[domain] = {}
        row: dict[str, Any] = {"domain": domain}
        for method in methods:
            averaged = average_query_metrics(
                _load_method_reports(run_root, domain, method)
            )
            query_values[domain][method] = averaged
            row[method] = {
                metric: fmean(value[metric] for value in averaged.values())
                for metric in METRICS
            }
        key_sets = {
            tuple(sorted(query_values[domain][method])) for method in methods
        }
        if len(key_sets) != 1:
            raise ValueError(f"paired query keys differ across methods for {domain}")
        domain_rows.append(row)

    macro = {
        method: {
            metric: fmean(row[method][metric] for row in domain_rows)
            for metric in METRICS
        }
        for method in methods
    }
    comparisons = {}
    for reference in ("baseline", "matched_random"):
        comparisons[f"ours_minus_{reference}"] = {}
        for metric in ("reciprocal_rank", "answer_reach_10"):
            differences = {}
            for domain in query_values:
                keys = sorted(query_values[domain]["ours"])
                differences[domain] = np.asarray(
                    [
                        query_values[domain]["ours"][key][metric]
                        - query_values[domain][reference][key][metric]
                        for key in keys
                    ],
                    dtype=np.float64,
                )
            estimate = paired_equal_domain_bootstrap(differences)
            comparisons[f"ours_minus_{reference}"][metric] = {
                **estimate,
                "difference_percentage_points": 100.0 * estimate["difference"],
                "ci_low_percentage_points": 100.0 * estimate["ci_low"],
                "ci_high_percentage_points": 100.0 * estimate["ci_high"],
            }
    return {"domains": domain_rows, "equal_domain_macro": macro, "comparisons": comparisons}


def aggregate_path_sensitivity(
    run_root: Path,
    domains: Iterable[str],
    variants: Iterable[int],
    seeds: Iterable[int],
) -> dict[str, Any]:
    rows = []
    for domain in domains:
        for method in ("baseline", "ours"):
            values = {"answer_path_mrr": [], "answer_reach_10": []}
            for variant in variants:
                for seed in seeds:
                    path = (
                        run_root
                        / "sensitivity"
                        / domain
                        / f"variant_{variant}"
                        / f"seed_{seed}"
                        / method
                        / "test_scores.report.json"
                    )
                    report = json.loads(path.read_text(encoding="utf-8"))
                    if (
                        report.get("evaluation_subset")
                        != "multiple_equal_shortest_paths"
                    ):
                        raise ValueError(
                            f"path sensitivity report has the wrong query subset: {path}"
                        )
                    values["answer_path_mrr"].append(
                        float(report["metrics"]["answer_path_mrr"])
                    )
                    values["answer_reach_10"].append(
                        float(report["metrics"]["answer_reach_10"])
                    )
            rows.append(
                {
                    "domain": domain,
                    "method": method,
                    **{
                        metric: {
                            "mean": fmean(metric_values),
                            "standard_deviation": pstdev(metric_values),
                            "minimum": min(metric_values),
                            "maximum": max(metric_values),
                            "range": max(metric_values) - min(metric_values),
                        }
                        for metric, metric_values in values.items()
                    },
                }
            )
    return {"domains": rows}
