"""从历史逐题 JSONL 流式生成 WWW 论文所需的 path-conflict 统计。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import numpy as np


def _distribution(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        raise ValueError("分布不能为空")
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "mean": float(array.mean()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "maximum": float(array.max()),
    }


def summarize_domain(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    totals = {
        "candidate_pool_size": 0,
        "disputed_candidate_count": 0,
        "sampled_negative_count": 0,
        "disputed_sampled_negative_count": 0,
    }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sampling = record["sampling"]
            sampled_counts = [int(item["sampled_count"]) for item in sampling]
            disputed_counts = [
                int(item["disputed_sampled_count"]) for item in sampling
            ]
            row = {
                "domain": record["domain"],
                "query_index": int(record["query_index"]),
                "candidate_pool_size": int(record["candidate_pool_size"]),
                "disputed_candidate_count": int(record["disputed_candidate_count"]),
                "candidate_conflict_rate": float(record["disputed_candidate_rate"]),
                "sampled_negative_count_mean": fmean(sampled_counts),
                "disputed_sampled_count_mean": fmean(disputed_counts),
                "sampled_conflict_rate_mean": fmean(
                    float(item["disputed_sampled_rate"]) for item in sampling
                ),
                "has_candidate_conflict": int(record["disputed_candidate_count"] > 0),
                "has_sampled_conflict_any_seed": int(any(disputed_counts)),
                "sampled_conflict_seed_fraction": sum(value > 0 for value in disputed_counts)
                / len(disputed_counts),
            }
            rows.append(row)
            totals["candidate_pool_size"] += row["candidate_pool_size"]
            totals["disputed_candidate_count"] += row["disputed_candidate_count"]
            totals["sampled_negative_count"] += sum(sampled_counts)
            totals["disputed_sampled_negative_count"] += sum(disputed_counts)
    if not rows:
        raise ValueError(f"audit JSONL 为空：{path}")
    domains = {row["domain"] for row in rows}
    if len(domains) != 1:
        raise ValueError(f"单个 audit JSONL 混入多个领域：{path}")
    query_count = len(rows)
    summary = {
        "domain": next(iter(domains)),
        "query_count": query_count,
        **totals,
        "candidate_conflict_rate": totals["disputed_candidate_count"]
        / totals["candidate_pool_size"],
        "sampled_conflict_rate": totals["disputed_sampled_negative_count"]
        / totals["sampled_negative_count"],
        "queries_with_candidate_conflict": sum(
            row["has_candidate_conflict"] for row in rows
        ),
        "queries_with_candidate_conflict_rate": fmean(
            row["has_candidate_conflict"] for row in rows
        ),
        "queries_with_sampled_conflict_any_seed": sum(
            row["has_sampled_conflict_any_seed"] for row in rows
        ),
        "queries_with_sampled_conflict_any_seed_rate": fmean(
            row["has_sampled_conflict_any_seed"] for row in rows
        ),
        "candidate_conflicts_per_query": _distribution(
            row["disputed_candidate_count"] for row in rows
        ),
        "sampled_conflicts_per_query_seed_average": _distribution(
            row["disputed_sampled_count_mean"] for row in rows
        ),
    }
    return summary, rows


def aggregate(
    input_root: Path,
    historical_summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    historical = json.loads(historical_summary_path.read_text(encoding="utf-8"))
    expected_domains = [row["domain"] for row in historical["domains"]]
    summaries = []
    per_query = []
    for domain in expected_domains:
        summary, rows = summarize_domain(input_root / f"{domain}.audit.jsonl")
        summaries.append(summary)
        per_query.extend(rows)

    overall = {
        "domain_count": len(summaries),
        "query_count": sum(row["query_count"] for row in summaries),
        "candidate_pool_size": sum(row["candidate_pool_size"] for row in summaries),
        "disputed_candidate_count": sum(
            row["disputed_candidate_count"] for row in summaries
        ),
        "sampled_negative_count": sum(
            row["sampled_negative_count"] for row in summaries
        ),
        "disputed_sampled_negative_count": sum(
            row["disputed_sampled_negative_count"] for row in summaries
        ),
        "queries_with_candidate_conflict": sum(
            row["queries_with_candidate_conflict"] for row in summaries
        ),
        "queries_with_sampled_conflict_any_seed": sum(
            row["queries_with_sampled_conflict_any_seed"] for row in summaries
        ),
    }
    overall["candidate_conflict_rate"] = (
        overall["disputed_candidate_count"] / overall["candidate_pool_size"]
    )
    overall["sampled_conflict_rate"] = (
        overall["disputed_sampled_negative_count"]
        / overall["sampled_negative_count"]
    )
    overall["queries_with_candidate_conflict_rate"] = (
        overall["queries_with_candidate_conflict"] / overall["query_count"]
    )
    overall["queries_with_sampled_conflict_any_seed_rate"] = (
        overall["queries_with_sampled_conflict_any_seed"] / overall["query_count"]
    )
    overall["candidate_conflicts_per_query"] = _distribution(
        row["disputed_candidate_count"] for row in per_query
    )
    overall["sampled_conflicts_per_query_seed_average"] = _distribution(
        row["disputed_sampled_count_mean"] for row in per_query
    )

    expected = historical["overall"]
    exact_pairs = {
        "query_count": "query_count",
        "candidate_pool_size": "candidate_count",
        "disputed_candidate_count": "disputed_candidate_count",
        "sampled_negative_count": "sampled_count",
        "disputed_sampled_negative_count": "disputed_sampled_count",
    }
    mismatches = {
        new: (overall[new], expected[old])
        for new, old in exact_pairs.items()
        if overall[new] != expected[old]
    }
    if mismatches:
        raise ValueError(f"逐题 audit 与历史汇总不一致：{mismatches}")

    report = {
        "schema_version": 1,
        "source": "historical_raw_audit_jsonl",
        "historical_totals_verified": True,
        "overall": overall,
        "domains": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "path_conflict_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "path_conflict_by_domain.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "domain",
            "query_count",
            "candidate_pool_size",
            "disputed_candidate_count",
            "candidate_conflict_rate",
            "sampled_negative_count",
            "disputed_sampled_negative_count",
            "sampled_conflict_rate",
            "queries_with_candidate_conflict",
            "queries_with_candidate_conflict_rate",
            "queries_with_sampled_conflict_any_seed",
            "queries_with_sampled_conflict_any_seed_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "path_conflict_per_query.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--historical-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = aggregate(args.input_root, args.historical_summary, args.output_dir)
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
