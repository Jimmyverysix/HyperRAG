from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research.path_consistent_negative_learning.scripts.build_retriever_manifest import (
    build_jobs,
)


class RetrieverManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "domains": ["a", "b"],
            "training": {"seeds": [42, 43]},
            "selection": {"lambda_grid": [0.0, 1.0]},
            "retrieval": {"beam_width": 10},
            "path_selection_sensitivity": {"variant_seeds": [2718, 3141, 5772]},
        }
        self.arguments = {
            "structured_root": Path("/structured"),
            "nlg_root": Path("/nlg"),
            "label_snapshot": Path("/labels.json"),
            "model_path": Path("/gte"),
            "run_root": Path("/runs"),
        }

    def test_phase_counts(self) -> None:
        self.assertEqual(len(build_jobs("encode", self.config, **self.arguments)), 2)
        self.assertEqual(len(build_jobs("prepare", self.config, **self.arguments)), 6)
        self.assertEqual(len(build_jobs("sweep", self.config, **self.arguments)), 8)

    def test_domain_subset_preserves_requested_order(self) -> None:
        jobs = build_jobs(
            "encode", self.config, domains=["b", "a"], **self.arguments
        )
        self.assertEqual([job["job_id"] for job in jobs], ["encode-b", "encode-a"])

    def test_domain_subset_rejects_domain_outside_frozen_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "not present in the frozen config"):
            build_jobs("encode", self.config, domains=["c"], **self.arguments)

    def test_domain_subset_rejects_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            build_jobs("encode", self.config, domains=["a", "a"], **self.arguments)

    def test_formal_manifest_rejects_pending_beam_decision(self) -> None:
        pending = {**self.config, "status": "p0_beam_selection_pending"}
        with self.assertRaisesRegex(ValueError, "beam P0 decision"):
            build_jobs("prepare", pending, **self.arguments)
        self.assertEqual(len(build_jobs("encode", pending, **self.arguments)), 2)

    def test_sweep_job_trains_then_evaluates_valid(self) -> None:
        job = build_jobs("sweep", self.config, **self.arguments)[0]
        self.assertEqual(len(job["commands"]), 2)
        self.assertIn("train", job["commands"][0])
        self.assertIn("evaluate", job["commands"][1])
        self.assertTrue(any("valid.pt" in token for token in job["commands"][1]))

    def test_post_selection_phase_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            selection_file = Path(temporary) / "selection.json"
            selection_file.write_text(
                json.dumps(
                    {
                        "domains": {
                            "a": {"lambda": 0.25},
                            "b": {"lambda": 0.0},
                        }
                    }
                ),
                encoding="utf-8",
            )
            arguments = {**self.arguments, "selection_file": selection_file}
            self.assertEqual(
                len(build_jobs("prepare-test", self.config, **arguments)), 2
            )
            self.assertEqual(
                len(build_jobs("main-test", self.config, **arguments)), 12
            )
            self.assertEqual(
                len(build_jobs("path-sensitivity", self.config, **arguments)), 12
            )
            self.assertEqual(
                len(
                    build_jobs(
                        "path-sensitivity-prepare", self.config, **arguments
                    )
                ),
                6,
            )
            self.assertEqual(
                len(
                    build_jobs(
                        "path-sensitivity-train", self.config, **arguments
                    )
                ),
                24,
            )
            main_job = build_jobs("main-test", self.config, **arguments)[0]
            sensitivity_job = build_jobs(
                "path-sensitivity", self.config, **arguments
            )[0]
            self.assertNotIn("--multiple-shortest-only", main_job["commands"][-1])
            self.assertIn(
                "--multiple-shortest-only", sensitivity_job["commands"][-1]
            )
            prepare_job = build_jobs(
                "path-sensitivity-prepare", self.config, **arguments
            )[0]
            train_job = build_jobs(
                "path-sensitivity-train", self.config, **arguments
            )[0]
            self.assertIn("--seeds", prepare_job["commands"][0])
            self.assertIn("--outputs", prepare_job["commands"][0])
            self.assertNotIn("prepare-train", train_job["commands"][0])
            self.assertIn("train", train_job["commands"][0])


if __name__ == "__main__":
    unittest.main()
