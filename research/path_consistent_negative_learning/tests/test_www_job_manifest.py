import json
import tempfile
import unittest
from pathlib import Path

from research.path_consistent_negative_learning.scripts.build_www_job_manifest import (
    build_jobs,
)


class WwwJobManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        research_root = Path(__file__).resolve().parents[1]
        cls.config = json.loads(
            (research_root / "configs" / "www_revision" / "proxy_main.json").read_text(
                encoding="utf-8"
            )
        )

    def test_selection_manifest_is_full_dataset_seed_lambda_product(self) -> None:
        jobs = build_jobs(
            "selection", self.config, Path("data"), Path("runs"), None
        )
        expected = (
            len(self.config["datasets"])
            * len(self.config["randomness"]["sampler_and_training_seeds"])
            * len(self.config["lambda_grid"])
        )
        self.assertEqual(len(jobs), expected)
        self.assertEqual(len({job["experiment_id"] for job in jobs}), expected)
        self.assertTrue(
            all("selection" in job["command"] for job in jobs)
        )
        self.assertEqual(
            self.config["dataset"]["name"],
            "WikiTopics_QE",
        )
        self.assertEqual(
            self.config["model"]["backbone"],
            "StructuredRetriever",
        )

    def test_controls_use_each_dataset_frozen_lambda(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            lambda_root = Path(temporary_directory)
            for domain in self.config["datasets"]:
                target = lambda_root / domain / "lambda_star.json"
                target.parent.mkdir(parents=True)
                target.write_text(
                    json.dumps(
                        {
                            "dataset": domain,
                            "best_lambda": 0.25,
                            "test_metrics_accessed": False,
                        }
                    ),
                    encoding="utf-8",
                )
            jobs = build_jobs(
                "test-controls",
                self.config,
                Path("data"),
                Path("runs"),
                lambda_root,
            )
        expected = (
            len(self.config["datasets"])
            * len(self.config["randomness"]["sampler_and_training_seeds"])
            * 2
        )
        self.assertEqual(len(jobs), expected)
        random_jobs = [
            job for job in jobs if job["expected_result"]["strategy"] == "random_weighted"
        ]
        self.assertTrue(
            all(job["expected_result"]["disputed_negative_weight"] == 0.25 for job in random_jobs)
        )


if __name__ == "__main__":
    unittest.main()
