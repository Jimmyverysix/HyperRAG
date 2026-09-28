"""Freeze all domain lambdas from complete Retriever-only valid sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from research.path_consistent_negative_learning.retriever_only.selection import (
    select_all_domains,
)
from research.path_consistent_negative_learning.retriever_only.provenance import (
    collect_provenance,
)


REPOSITORY = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = select_all_domains(args.run_root, args.domains)
    result["provenance"] = collect_provenance(REPOSITORY, sys.argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
