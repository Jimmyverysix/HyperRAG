"""Write a deterministic real-data example for the Retriever-only paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from research.path_consistent_negative_learning.retriever_only.case_study import (
    extract_case_study,
)
from research.path_consistent_negative_learning.retriever_only.provenance import (
    collect_provenance,
)


REPOSITORY = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-root", type=Path, required=True)
    parser.add_argument("--nlg-root", type=Path, required=True)
    parser.add_argument("--label-snapshot", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = extract_case_study(
        args.structured_root,
        args.nlg_root,
        args.label_snapshot,
        domain=args.domain,
        seed=args.seed,
    )
    result["provenance"] = collect_provenance(REPOSITORY, sys.argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"case study -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
