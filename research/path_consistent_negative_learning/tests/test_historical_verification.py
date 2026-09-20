import json
import tempfile
import unittest
from pathlib import Path

from research.path_consistent_negative_learning.scripts.verify_historical_artifacts import (
    verify_historical_artifacts,
)


class HistoricalArtifactVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.research_root = Path(__file__).resolve().parents[1]
        cls.artifacts = cls.research_root / "artifacts"

    def test_real_historical_artifacts_are_internally_consistent(self) -> None:
        report = verify_historical_artifacts(self.artifacts)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["facts"]["domain_count"], 11)
        self.assertEqual(report["facts"]["query_count"], 90000)
        self.assertAlmostEqual(
            report["facts"]["lambda_zero_selected_pr_auc_delta"],
            -0.013187252040638014,
        )
        self.assertAlmostEqual(report["facts"]["historical_transfer_lambda"], 0.1)

    def test_changed_audit_total_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            for relative in (
                "audit/combined_summary.json",
                "gate_c/training_summary.json",
                "weighted/selection_summary.json",
                "weighted/confirmation_summary.json",
                "weighted/gate_d_summary.json",
                "weighted/locked_weight.json",
                "provenance.json",
            ):
                source = self.artifacts / relative
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())

            summary_path = temporary / "audit" / "combined_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["overall"]["sampled_count"] += 1
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "audit_sampled_count_is_domain_sum"):
                verify_historical_artifacts(temporary)


if __name__ == "__main__":
    unittest.main()
