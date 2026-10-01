"""Generate WWW LaTeX numbers and tables from Retriever-only aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.path_consistent_negative_learning.retriever_only.paper import (
    generate_paper_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--prevalence", type=Path, required=True)
    parser.add_argument("--main-test", type=Path, required=True)
    parser.add_argument("--fixed-masking", type=Path, required=True)
    parser.add_argument("--candidate-oracle", type=Path, required=True)
    parser.add_argument("--lambda-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = generate_paper_artifacts(
        args.preflight,
        args.selection,
        args.prevalence,
        args.main_test,
        args.fixed_masking,
        args.candidate_oracle,
        args.lambda_validation,
        args.output_dir,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
