import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.path_consistent_negative_learning.scripts.build_www_job_manifest import (
    build_jobs,
)
from research.path_consistent_negative_learning.scripts.run_experiment_queue import (
    _wait_for_available_gpus,
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

    def test_queue_waits_for_cuda_context_release_between_phases(self) -> None:
        module = "research.path_consistent_negative_learning.scripts.run_experiment_queue"
        with patch(
            f"{module}._available_gpus",
            side_effect=[(), (0, 1)],
        ) as available, patch(f"{module}.time.sleep") as sleep:
            result = _wait_for_available_gpus((0, 1), 512)
        self.assertEqual(result, (0, 1))
        self.assertEqual(available.call_count, 2)
        sleep.assert_called_once_with(5.0)


if __name__ == "__main__":
    unittest.main()
