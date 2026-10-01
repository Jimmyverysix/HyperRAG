"""Generate final-revision validation, oracle, and fixed-masking artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

from research.path_consistent_negative_learning.retriever_only.final_revision import (
    aggregate_fixed_masking,
    analyze_candidate_oracle,
    analyze_validation_lambdas,
)
from research.path_consistent_negative_learning.retriever_only.provenance import (
    collect_provenance,
)


REPOSITORY = Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["provenance"] = collect_provenance(REPOSITORY, sys.argv)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_lambda_csv(path: Path, payload: dict[str, Any]) -> None:
    domains = sorted(payload["rows"][0]["per_domain_apc_mrr"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["lambda", "equal_domain_macro_apc_mrr", *domains],
        )
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow(
                {
                    "lambda": row["lambda"],
                    "equal_domain_macro_apc_mrr": row[
                        "equal_domain_macro_apc_mrr"
                    ],
                    **row["per_domain_apc_mrr"],
                }
            )


def _write_oracle_csv(path: Path, payload: dict[str, Any]) -> None:
    fields = [
        "scope",
        "domain",
        "query_count",
        "candidate_count",
        "oracle_reachable_query_count",
        "candidate_oracle_reach",
        "mean_candidate_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["domains"]:
            writer.writerow({"scope": "domain", **row})
        writer.writerow(
            {
                "scope": "equal_domain_macro",
                "domain": "ALL",
                "query_count": payload["overall_query_count"],
                "oracle_reachable_query_count": payload[
                    "overall_reachable_query_count"
                ],
                "candidate_oracle_reach": payload[
                    "equal_domain_macro_candidate_oracle_reach"
                ],
            }
        )


def _write_fixed_csv(path: Path, payload: dict[str, Any]) -> None:
    fields = [
        "row_type",
        "domain",
        "method",
        "metric",
        "value",
        "reference",
        "difference_percentage_points",
        "ci_low_percentage_points",
        "ci_high_percentage_points",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["domains"]:
            for method, metrics in row.items():
                if method == "domain":
                    continue
                for metric, value in metrics.items():
                    writer.writerow(
                        {
                            "row_type": "domain",
                            "domain": row["domain"],
                            "method": method,
                            "metric": metric,
                            "value": value,
                        }
                    )
        for method, metrics in payload["equal_domain_macro"].items():
            for metric, value in metrics.items():
                writer.writerow(
                    {
                        "row_type": "equal_domain_macro",
                        "domain": "ALL",
                        "method": method,
                        "metric": metric,
                        "value": value,
                    }
                )
        for comparison, metrics in payload["comparisons"].items():
            reference = comparison.removeprefix("fixed_masking_minus_")
            for metric, values in metrics.items():
                writer.writerow(
                    {
                        "row_type": "paired_bootstrap",
                        "domain": "ALL",
                        "method": "fixed_masking",
                        "metric": metric,
                        "reference": reference,
                        "difference_percentage_points": values[
                            "difference_percentage_points"
                        ],
                        "ci_low_percentage_points": values[
                            "ci_low_percentage_points"
                        ],
                        "ci_high_percentage_points": values[
                            "ci_high_percentage_points"
                        ],
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="analysis", required=True)
    validation = subparsers.add_parser("lambda-validation")
    validation.add_argument("--selection", type=Path, required=True)
    oracle = subparsers.add_parser("candidate-oracle")
    oracle.add_argument("--run-root", type=Path, required=True)
    oracle.add_argument("--domains", nargs="+", required=True)
    fixed = subparsers.add_parser("fixed-masking")
    fixed.add_argument("--run-root", type=Path, required=True)
    fixed.add_argument("--posthoc-root", type=Path, required=True)
    fixed.add_argument("--selection", type=Path, required=True)
    fixed.add_argument("--domains", nargs="+", required=True)
    for subparser in (validation, oracle, fixed):
        subparser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analysis == "lambda-validation":
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        payload = analyze_validation_lambdas(selection)
        _write_json(args.output_dir / "lambda_validation.json", payload)
        _write_lambda_csv(args.output_dir / "lambda_validation.csv", payload)
    elif args.analysis == "candidate-oracle":
        payload = analyze_candidate_oracle(args.run_root, args.domains)
        _write_json(args.output_dir / "candidate_oracle.json", payload)
        _write_oracle_csv(args.output_dir / "candidate_oracle.csv", payload)
    else:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        payload = aggregate_fixed_masking(
            args.run_root,
            args.posthoc_root,
            selection,
            args.domains,
        )
        _write_json(args.output_dir / "fixed_masking.json", payload)
        _write_fixed_csv(args.output_dir / "fixed_masking_results.csv", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
