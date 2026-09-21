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


def _write_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_all(
    config_path: Path,
    run_root: Path,
    output_root: Path,
    snapshot_root: Path | None = None,
) -> list[dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = []
    selections = {}
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
        selections[domain] = selection
        row = {
            "dataset": domain,
            "best_lambda": selection["best_lambda"],
            **{
                f"lambda_{value:g}": selection["validation_scores"][str(value)]["mean"]
                for value in DEFAULT_LAMBDA_GRID
            },
        }
        rows.append(row)
    _write_summary_csv(rows, output_root / "lambda_selection.csv")
    if snapshot_root is not None:
        domains_root = snapshot_root / "domains"
        domains_root.mkdir(parents=True, exist_ok=True)
        for domain, selection in selections.items():
            (domains_root / f"{domain}.json").write_text(
                json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        _write_summary_csv(rows, snapshot_root / "lambda_selection.csv")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = select_all(
        args.config,
        args.run_root,
        args.output_root,
        args.snapshot_root,
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
