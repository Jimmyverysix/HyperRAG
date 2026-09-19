"""Build one fixed-candidate WikiTopics dataset for a shared experiment seed."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .structured_data import build_structured_experiment_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-dir", type=Path, required=True)
    parser.add_argument("--sampler-seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=20260919)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = build_structured_experiment_data(
        args.domain_dir,
        sampler_seed=args.sampler_seed,
        split_seed=args.split_seed,
    )
    data.save(args.output)
    print(
        f"saved {data.domain}: {len(data.query_keys)} queries, "
        f"{len(data.selected_labels)} candidates -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
