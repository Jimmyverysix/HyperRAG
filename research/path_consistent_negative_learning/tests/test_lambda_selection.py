import json
import tempfile
import unittest
from pathlib import Path

from research.path_consistent_negative_learning.path_supervision.lambda_selection import (
    DEFAULT_LAMBDA_GRID,
    select_lambda,
)
from research.path_consistent_negative_learning.scripts.select_all_lambdas import (
    select_all,
)


def _result(lambda_value, seed, score, split="selection"):
    return {
        "domain": "toy",
        "strategy": "strategy2_weighted",
        "seed": seed,
        "evaluation_split": split,
        "disputed_negative_weight": lambda_value,
        "metrics": {
            "answer_reach_at": {"10": score},
            "selected_path": {"mrr": score, "question_pr_auc": score},
            "all_shortest_paths": {"mrr": score, "question_pr_auc": score},
        },
    }


class LambdaSelectionTests(unittest.TestCase):
    def test_uses_all_grid_points_and_shared_seeds(self) -> None:
        records = []
        for lambda_value in DEFAULT_LAMBDA_GRID:
            for seed in (42, 43):
                score = 0.8 - abs(lambda_value - 0.25)
                records.append(_result(lambda_value, seed, score))

        selected = select_lambda(records, selection_metric="answer_reach_10")

        self.assertEqual(selected["best_lambda"], 0.25)
        self.assertFalse(selected["test_metrics_accessed"])
        self.assertEqual(selected["seeds"], [42, 43])

    def test_rejects_test_results(self) -> None:
        records = [
            _result(lambda_value, 42, 0.8, split="test")
            for lambda_value in DEFAULT_LAMBDA_GRID
        ]
        with self.assertRaisesRegex(ValueError, "test leakage"):
            select_lambda(records, selection_metric="answer_reach_10")

    def test_exact_tie_prefers_weaker_intervention(self) -> None:
        records = [
            _result(lambda_value, 42, 0.9)
            for lambda_value in DEFAULT_LAMBDA_GRID
        ]
        selected = select_lambda(records, selection_metric="answer_reach_10")
        self.assertEqual(selected["best_lambda"], 1.0)

    def test_all_domain_selection_writes_repository_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "datasets": ["toy"],
                        "lambda_selection": {"metric": "answer_reach_10"},
                    }
                ),
                encoding="utf-8",
            )
            run_root = root / "runs"
            for lambda_value in DEFAULT_LAMBDA_GRID:
                for seed in (42, 43):
                    result_path = (
                        run_root
                        / "selection"
                        / "toy"
                        / f"strategy2_weight_{lambda_value:g}"
                        / f"seed_{seed}"
                        / "result.json"
                    )
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(
                        json.dumps(_result(lambda_value, seed, 1.0 - lambda_value)),
                        encoding="utf-8",
                    )

            output_root = root / "selection"
            snapshot_root = root / "artifacts" / "selection"
            select_all(config, run_root, output_root, snapshot_root)

            self.assertEqual(
                json.loads((output_root / "toy" / "lambda_star.json").read_text()),
                json.loads((snapshot_root / "domains" / "toy.json").read_text()),
            )
            self.assertEqual(
                (output_root / "lambda_selection.csv").read_text(),
                (snapshot_root / "lambda_selection.csv").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
