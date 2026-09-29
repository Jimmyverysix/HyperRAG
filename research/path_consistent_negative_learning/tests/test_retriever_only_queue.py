import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.path_consistent_negative_learning.scripts.run_retriever_queue import (
    run_manifest,
)


class RetrieverQueueTests(unittest.TestCase):
    def test_rejects_nonpositive_workers_per_gpu_before_reading_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "workers_per_gpu"):
            run_manifest(
                Path("does-not-exist.json"),
                gpus=(1,),
                maximum_used_mib=512,
                workers_per_gpu=0,
            )

    def test_repeats_each_available_gpu_as_an_independent_slot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = []
            for index in range(8):
                jobs.append(
                    {
                        "job_id": f"job-{index}",
                        "output_dir": str(root / f"job-{index}"),
                        "commands": [["{python}", "-c", "pass"]],
                        "expected_files": [],
                    }
                )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"phase": "test", "jobs": jobs}),
                encoding="utf-8",
            )

            with (
                patch(
                    "research.path_consistent_negative_learning.scripts."
                    "run_retriever_queue._wait_for_gpus",
                    return_value=(1, 2),
                ),
                patch(
                    "research.path_consistent_negative_learning.scripts."
                    "run_retriever_queue.subprocess.run"
                ) as run,
            ):
                run.return_value.returncode = 0
                run_manifest(
                    manifest,
                    gpus=(1, 2),
                    maximum_used_mib=512,
                    workers_per_gpu=4,
                )

            self.assertEqual(run.call_count, 8)
            reports = [
                json.loads(
                    (root / f"job-{index}" / "queue_provenance.json").read_text(
                        encoding="utf-8"
                    )
                )
                for index in range(8)
            ]
            self.assertEqual({report["physical_gpu_id"] for report in reports}, {1, 2})
            self.assertTrue(all(report["workers_per_gpu"] == 4 for report in reports))
            self.assertTrue((root / "manifest.complete").is_file())


if __name__ == "__main__":
    unittest.main()
