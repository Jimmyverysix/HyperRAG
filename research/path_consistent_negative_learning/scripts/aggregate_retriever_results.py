"""Create formal aggregate JSON files from completed Retriever-only runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.path_consistent_negative_learning.retriever_only.aggregation import (
    aggregate_conflict_prevalence,
    aggregate_main_test,
    aggregate_path_sensitivity,
)


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
    return parser


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
    result.update({"schema_version": 1, "phase": args.phase})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{args.phase} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
