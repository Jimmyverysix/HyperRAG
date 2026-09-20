import csv
import tempfile
import unittest
from pathlib import Path

from research.path_consistent_negative_learning.scripts.analyze_semantic_audit import (
    analyze,
)


class SemanticAuditTests(unittest.TestCase):
    def _write(self, path: Path, labels: list[tuple[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("item_id", "domain", "semantic_label")
            )
            writer.writeheader()
            for item_id, label in labels:
                writer.writerow(
                    {"item_id": item_id, "domain": "toy", "semantic_label": label}
                )

    def test_reports_disagreement_without_automatic_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            a = root / "a.csv"
            b = root / "b.csv"
            self._write(a, [("x", "relevant"), ("y", "irrelevant")])
            self._write(b, [("x", "relevant"), ("y", "unclear")])

            report = analyze(a, b, root / "out")

            self.assertEqual(report["raw_agreement"], 0.5)
            self.assertEqual(report["disagreement_count"], 1)
            self.assertFalse(report["automatic_consensus_used"])
            self.assertEqual(report["status"], "awaiting_human_consensus")

    def test_rejects_unfilled_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            a = root / "a.csv"
            b = root / "b.csv"
            self._write(a, [("x", "")])
            self._write(b, [("x", "relevant")])

            with self.assertRaisesRegex(ValueError, "空标签"):
                analyze(a, b, root / "out")


if __name__ == "__main__":
    unittest.main()
