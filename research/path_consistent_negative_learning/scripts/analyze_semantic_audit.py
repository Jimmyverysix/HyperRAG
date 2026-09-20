"""分析两名真实标注者填写的语义审计表，不自动裁决分歧。"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


LABELS = ("relevant", "irrelevant", "unclear")


def _load(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {row["item_id"]: row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"存在重复 item_id：{path}")
    invalid = {
        row["semantic_label"]
        for row in rows
        if row["semantic_label"] not in LABELS
    }
    if invalid:
        raise ValueError(f"{path} 含空标签或非法标签：{sorted(invalid)}")
    return keyed


def analyze(annotator_a: Path, annotator_b: Path, output_dir: Path) -> dict:
    a = _load(annotator_a)
    b = _load(annotator_b)
    if set(a) != set(b):
        raise ValueError("两名标注者的 item_id 集合不一致")
    item_ids = sorted(a)
    labels_a = [a[item]["semantic_label"] for item in item_ids]
    labels_b = [b[item]["semantic_label"] for item in item_ids]
    disagreements = [
        {
            "item_id": item,
            "domain": a[item]["domain"],
            "annotator_a": a[item]["semantic_label"],
            "annotator_b": b[item]["semantic_label"],
            "consensus_label": "",
        }
        for item in item_ids
        if a[item]["semantic_label"] != b[item]["semantic_label"]
    ]
    agreement = 1.0 - len(disagreements) / len(item_ids)
    report = {
        "schema_version": 1,
        "status": "awaiting_human_consensus" if disagreements else "complete",
        "item_count": len(item_ids),
        "labels": list(LABELS),
        "annotator_a_counts": dict(Counter(labels_a)),
        "annotator_b_counts": dict(Counter(labels_b)),
        "raw_agreement": agreement,
        "cohen_kappa": float(cohen_kappa_score(labels_a, labels_b, labels=LABELS)),
        "disagreement_count": len(disagreements),
        "automatic_consensus_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "agreement.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "disagreements.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "item_id",
                "domain",
                "annotator_a",
                "annotator_b",
                "consensus_label",
            ),
        )
        writer.writeheader()
        writer.writerows(disagreements)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = analyze(args.annotator_a, args.annotator_b, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
