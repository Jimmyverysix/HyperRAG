"""Create formal aggregate JSON files from completed Retriever-only runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

from research.path_consistent_negative_learning.retriever_only.aggregation import (
    aggregate_conflict_prevalence,
    aggregate_main_test,
    aggregate_path_sensitivity,
)
from research.path_consistent_negative_learning.retriever_only.provenance import (
    collect_provenance,
)


REPOSITORY = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("prevalence", "main-test", "sensitivity"), required=True
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--variants", type=int, nargs="+", default=[2718, 3141, 5772])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    return parser


def _csv_rows(
    phase: str,
    result: dict[str, Any],
) -> tuple[list[str], Iterable[dict[str, Any]]]:
    if phase == "prevalence":
        fields = [
            "domain",
            "sampled_negative_count",
            "path_consistent_negative_count",
            "path_consistent_negative_rate",
            "query_count",
            "affected_query_count_across_seeds",
            "affected_query_rate_across_seeds",
        ]
        return fields, (
            {field: row[field] for field in fields} for row in result["domains"]
        )
    if phase == "main-test":
        fields = [
            "scope",
            "domain",
            "method",
            "answer_path_mrr",
            "answer_reach_10",
            "answer_reach_5",
        ]
        rows = []
        metric_names = {
            "answer_path_mrr": "reciprocal_rank",
            "answer_reach_10": "answer_reach_10",
            "answer_reach_5": "answer_reach_5",
        }
        for domain_row in result["domains"]:
            for method in ("baseline", "matched_random", "ours"):
                rows.append(
                    {
                        "scope": "domain",
                        "domain": domain_row["domain"],
                        "method": method,
                        **{
                            output: domain_row[method][source]
                            for output, source in metric_names.items()
                        },
                    }
                )
        for method, metrics in result["equal_domain_macro"].items():
            rows.append(
                {
                    "scope": "equal_domain_macro",
                    "domain": "ALL",
                    "method": method,
                    **{
                        output: metrics[source]
                        for output, source in metric_names.items()
                    },
                }
            )
        return fields, rows
    fields = [
        "domain",
        "method",
        "query_count",
        "metric",
        "mean",
        "standard_deviation",
        "minimum",
        "maximum",
        "range",
    ]
    rows = []
    for row in result["domains"]:
        for metric in ("answer_path_mrr", "answer_reach_10"):
            rows.append(
                {
                    "domain": row["domain"],
                    "method": row["method"],
                    "query_count": row["query_count"],
                    "metric": metric,
                    **row[metric],
                }
            )
    return fields, rows


def _write_csv(path: Path, phase: str, result: dict[str, Any]) -> None:
    fields, rows = _csv_rows(phase, result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    if args.phase == "prevalence":
        result = aggregate_conflict_prevalence(args.run_root, args.domains, args.seeds)
    elif args.phase == "main-test":
        result = aggregate_main_test(args.run_root, args.domains)
    else:
        result = aggregate_path_sensitivity(
            args.run_root, args.domains, args.variants, args.seeds
        )
    result.update(
        {
            "schema_version": 1,
            "phase": args.phase,
            "provenance": collect_provenance(REPOSITORY, sys.argv),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_output = args.csv_output or args.output.with_suffix(".csv")
    _write_csv(csv_output, args.phase, result)
    print(f"{args.phase} -> {args.output}, {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
