"""为所有 dataset 从完整 selection sweep 冻结各自的 lambda。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from research.path_consistent_negative_learning.path_supervision.lambda_selection import (
    DEFAULT_LAMBDA_GRID,
    load_selection_results,
    select_lambda,
)


def select_all(config_path: Path, run_root: Path, output_root: Path) -> list[dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = []
    for domain in config["datasets"]:
        paths = sorted(
            (run_root / "selection" / domain).glob(
                "strategy2_weight_*/seed_*/result.json"
            )
        )
        selection = select_lambda(
            load_selection_results(paths),
            selection_metric=config["lambda_selection"]["metric"],
            lambda_grid=DEFAULT_LAMBDA_GRID,
        )
        target = output_root / domain / "lambda_star.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        row = {
            "dataset": domain,
            "best_lambda": selection["best_lambda"],
            **{
                f"lambda_{value:g}": selection["validation_scores"][str(value)]["mean"]
                for value in DEFAULT_LAMBDA_GRID
            },
        }
        rows.append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "lambda_selection.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = select_all(args.config, args.run_root, args.output_root)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
