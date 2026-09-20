"""从完整 validation sweep 自动生成冻结的 ``lambda_star.json``。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.path_consistent_negative_learning.path_supervision.lambda_selection import (
    DEFAULT_LAMBDA_GRID,
    load_selection_results,
    select_lambda,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        required=True,
        help="包含 strategy2_weight_*/seed_*/result.json 的单数据集目录",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-metric", default="answer_reach_10")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = sorted(args.result_root.glob("strategy2_weight_*/seed_*/result.json"))
    results = load_selection_results(paths)
    selection = select_lambda(
        results,
        selection_metric=args.selection_metric,
        lambda_grid=DEFAULT_LAMBDA_GRID,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{selection['dataset']}: lambda*={selection['best_lambda']} "
        f"({selection['selection_metric']}) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
