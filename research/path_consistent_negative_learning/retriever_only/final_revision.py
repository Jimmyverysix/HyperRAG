"""Minimal analyses required for the final Retriever-only revision."""

from __future__ import annotations

from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

import numpy as np

from .aggregation import (
    FORMAL_SEEDS,
    METRICS,
    average_query_metrics,
    paired_equal_domain_bootstrap,
)
from .evaluation import candidate_path_coverage
from .prepared import PreparedCandidates


def analyze_validation_lambdas(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Compute domain-equal validation APC-MRR for every frozen lambda."""

    if selection.get("test_metrics_accessed") is not False:
        raise ValueError("lambda simplification accepts only pre-test validation data")
    domains = selection["domains"]
    rows = []
    for lambda_ in selection["lambda_grid"]:
        key = str(float(lambda_))
        per_domain = {
            domain: float(value["validation"][key]["mean_answer_path_mrr"])
            for domain, value in domains.items()
        }
        rows.append(
            {
                "lambda": float(lambda_),
                "equal_domain_macro_apc_mrr": fmean(per_domain.values()),
                "per_domain_apc_mrr": per_domain,
            }
        )
    best_value = max(row["equal_domain_macro_apc_mrr"] for row in rows)
    best_lambda = max(
        row["lambda"]
        for row in rows
        if abs(row["equal_domain_macro_apc_mrr"] - best_value) <= 1e-12
    )
    return {
        "schema_version": 1,
        "analysis_split": "valid",
        "metric": "APC-MRR",
        "aggregation": "equal-domain macro after five-seed domain means",
        "rows": rows,
        "best_global_lambda": best_lambda,
        "test_metrics_accessed": False,
    }


def analyze_candidate_oracle(
    run_root: Path,
    domains: Iterable[str],
) -> dict[str, Any]:
    """Measure complete-path reachability in the frozen test candidate pools."""

    rows = []
    for domain in domains:
        data = PreparedCandidates.load(
            run_root / "prepared" / "eval" / domain / "test.pt"
        )
        if data.domain != domain or data.split != "test":
            raise ValueError(f"candidate pool metadata mismatch for {domain}")
        coverage = candidate_path_coverage(data)
        rows.append(
            {
                "domain": domain,
                "query_count": len(data.query_keys),
                "candidate_count": len(data.dde_features),
                "oracle_reachable_query_count": coverage[
                    "path_reachable_query_count"
                ],
                "candidate_oracle_reach": coverage["path_reachable_query_rate"],
                "mean_candidate_count": coverage["mean_candidate_count"],
            }
        )
    total_queries = sum(row["query_count"] for row in rows)
    total_reachable = sum(row["oracle_reachable_query_count"] for row in rows)
    return {
        "schema_version": 1,
        "split": "test",
        "beam_width": 32,
        "definition": "complete directed topic-to-answer path exists in fixed pool",
        "domains": rows,
        "equal_domain_macro_candidate_oracle_reach": fmean(
            row["candidate_oracle_reach"] for row in rows
        ),
        "overall_query_count": total_queries,
        "overall_reachable_query_count": total_reachable,
        "micro_candidate_oracle_reach": total_reachable / total_queries,
    }


def _load_reports(
    directory: Path,
    *,
    expected_methods: set[str],
) -> list[dict[str, Any]]:
    import json

    paths = sorted(directory.glob("seed_*/test_scores.report.json"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if {int(report["seed"]) for report in reports} != set(FORMAL_SEEDS):
        raise ValueError(f"incomplete formal seed set in {directory}")
    if any(report.get("split") != "test" for report in reports):
        raise ValueError(f"non-test report in {directory}")
    if any(report.get("method") not in expected_methods for report in reports):
        raise ValueError(f"method metadata mismatch in {directory}")
    return reports


def aggregate_fixed_masking(
    run_root: Path,
    posthoc_root: Path,
    selection: Mapping[str, Any],
    domains: Iterable[str],
) -> dict[str, Any]:
    """Aggregate the explicitly post-hoc global-lambda-zero comparison."""

    methods = (
        "baseline",
        "matched_random_masking",
        "fixed_masking",
        "tuned_strategy",
    )
    query_values: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    domain_rows = []
    sources: dict[str, dict[str, str]] = {}
    for domain in domains:
        selected_lambda = float(selection["domains"][domain]["lambda"])
        main = run_root / "test" / domain
        posthoc = posthoc_root / domain
        source_specs = {
            "baseline": (main / "baseline", {"baseline"}),
            "tuned_strategy": (main / "ours", {"ours"}),
        }
        if selected_lambda == 0.0:
            source_specs.update(
                {
                    "matched_random_masking": (
                        main / "matched_random",
                        {"matched_random"},
                    ),
                    "fixed_masking": (main / "ours", {"ours"}),
                }
            )
        else:
            source_specs.update(
                {
                    "matched_random_masking": (
                        posthoc / "matched_random_masking",
                        {"fixed_matched_random"},
                    ),
                    "fixed_masking": (
                        posthoc / "fixed_masking",
                        {"fixed_masking"},
                    ),
                }
            )
        query_values[domain] = {}
        row: dict[str, Any] = {"domain": domain}
        sources[domain] = {}
        for method, (directory, expected) in source_specs.items():
            averaged = average_query_metrics(
                _load_reports(directory, expected_methods=expected)
            )
            query_values[domain][method] = averaged
            row[method] = {
                metric: fmean(value[metric] for value in averaged.values())
                for metric in METRICS
            }
            sources[domain][method] = str(directory)
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
    domain_comparisons = []
    for reference in ("baseline", "matched_random_masking", "tuned_strategy"):
        label = f"fixed_masking_minus_{reference}"
        comparisons[label] = {}
        for metric in ("reciprocal_rank", "answer_reach_10"):
            differences = {}
            for domain in query_values:
                keys = sorted(query_values[domain]["fixed_masking"])
                differences[domain] = np.asarray(
                    [
                        query_values[domain]["fixed_masking"][key][metric]
                        - query_values[domain][reference][key][metric]
                        for key in keys
                    ],
                    dtype=np.float64,
                )
            estimate = paired_equal_domain_bootstrap(differences)
            comparisons[label][metric] = {
                **estimate,
                "difference_percentage_points": 100.0 * estimate["difference"],
                "ci_low_percentage_points": 100.0 * estimate["ci_low"],
                "ci_high_percentage_points": 100.0 * estimate["ci_high"],
            }
            if metric == "reciprocal_rank":
                for domain, values in differences.items():
                    domain_estimate = paired_equal_domain_bootstrap(
                        {domain: values}
                    )
                    domain_comparisons.append(
                        {
                            "domain": domain,
                            "reference": reference,
                            "metric": "APC-MRR",
                            "difference_percentage_points": 100.0
                            * domain_estimate["difference"],
                            "ci_low_percentage_points": 100.0
                            * domain_estimate["ci_low"],
                            "ci_high_percentage_points": 100.0
                            * domain_estimate["ci_high"],
                            "resamples": domain_estimate["resamples"],
                            "seed": domain_estimate["seed"],
                        }
                    )
    final_main = {
        "domains": [
            {
                "domain": row["domain"],
                "baseline": row["baseline"],
                "matched_random": row["matched_random_masking"],
                "ours": row["fixed_masking"],
            }
            for row in domain_rows
        ],
        "equal_domain_macro": {
            "baseline": macro["baseline"],
            "matched_random": macro["matched_random_masking"],
            "ours": macro["fixed_masking"],
        },
        "comparisons": {
            "ours_minus_baseline": comparisons["fixed_masking_minus_baseline"],
            "ours_minus_matched_random": comparisons[
                "fixed_masking_minus_matched_random_masking"
            ],
        },
        "domain_comparisons": [
            {
                **row,
                "reference": (
                    "matched_random"
                    if row["reference"] == "matched_random_masking"
                    else row["reference"]
                ),
            }
            for row in domain_comparisons
            if row["reference"] in {"baseline", "matched_random_masking"}
        ],
    }
    return {
        "schema_version": 1,
        "analysis_design": "POST-HOC SIMPLIFICATION ANALYSIS",
        "global_lambda": 0.0,
        "domains": domain_rows,
        "equal_domain_macro": macro,
        "comparisons": comparisons,
        "domain_comparisons": domain_comparisons,
        "final_main": final_main,
        "sources": sources,
    }
