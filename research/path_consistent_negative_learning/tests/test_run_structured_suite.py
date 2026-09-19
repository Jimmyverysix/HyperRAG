from pathlib import Path
import tempfile
import unittest

from research.path_consistent_negative_learning.run_structured_suite import build_tasks


class StructuredSuiteTests(unittest.TestCase):
    def test_builds_four_tasks_per_seed_with_independent_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            for seed in (42, 43):
                (data_dir / f"seed_{seed}.pt").touch()
            tasks = build_tasks(data_dir, root / "runs", (42, 43))
        self.assertEqual(len(tasks), 8)
        self.assertEqual(len({task.output_dir for task in tasks}), 8)
        self.assertEqual({task.seed for task in tasks}, {42, 43})


if __name__ == "__main__":
    unittest.main()
