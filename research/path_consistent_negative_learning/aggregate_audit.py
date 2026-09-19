"""Aggregate WikiTopics domain summaries and make the preregistered Gate B decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .reporting import aggregate_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = aggregate_directory(args.input_dir, args.output_dir)
    print(json.dumps(report["gate_b"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
