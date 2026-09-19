import math
import unittest

from research.path_consistent_negative_learning.statistics import (
    DuplicateKeyError,
    MissingKeyError,
    PairingError,
    pair_by_key,
    paired_bootstrap_mean_difference,
    summarize_paired_seeds,
)


class PairedStatisticsTests(unittest.TestCase):
    def test_pairing_uses_query_keys_not_input_position(self) -> None:
        paired = pair_by_key(
            [("q1", 1.0), ("q2", 3.0)],
            [("q2", 5.0), ("q1", 2.0)],
        )
        self.assertEqual(paired.keys, ("q1", "q2"))
        self.assertEqual(paired.reference.tolist(), [1.0, 3.0])
        self.assertEqual(paired.comparison.tolist(), [2.0, 5.0])

    def test_duplicate_and_missing_query_keys_are_errors(self) -> None:
        with self.assertRaisesRegex(DuplicateKeyError, "duplicate key"):
            pair_by_key([("q1", 1.0), ("q1", 2.0)], {"q1": 1.0})
        with self.assertRaisesRegex(MissingKeyError, "missing from comparison"):
            pair_by_key({"q1": 1.0, "q2": 2.0}, {"q1": 1.0})
        with self.assertRaisesRegex(MissingKeyError, "missing from reference"):
            pair_by_key({"q1": 1.0}, {"q1": 1.0, "q2": 2.0})

    def test_bootstrap_is_paired_and_reproducible(self) -> None:
        reference = {"q1": 1.0, "q2": 3.0, "q3": 2.0}
        comparison = {"q1": 2.0, "q2": 2.0, "q3": 4.0}
        first = paired_bootstrap_mean_difference(
            reference,
            comparison,
            n_resamples=2_000,
            seed=17,
        )
        second = paired_bootstrap_mean_difference(
            reference,
            comparison,
            n_resamples=2_000,
            seed=17,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.mean_difference, 2.0 / 3.0)
        self.assertLessEqual(first.ci_low, first.mean_difference)
        self.assertGreaterEqual(first.ci_high, first.mean_difference)

    def test_constant_paired_difference_has_exact_interval(self) -> None:
        result = paired_bootstrap_mean_difference(
            {"a": 1.0, "b": 2.0},
            {"a": 1.5, "b": 2.5},
            n_resamples=100,
        )
        self.assertEqual(result.mean_difference, 0.5)
        self.assertEqual(result.ci_low, 0.5)
        self.assertEqual(result.ci_high, 0.5)

    def test_seed_summary_pairs_queries_within_each_seed(self) -> None:
        result = summarize_paired_seeds(
            {
                1: {"q1": 1.0, "q2": 2.0},
                2: {"q1": 2.0, "q2": 4.0},
            },
            {
                1: [("q2", 4.0), ("q1", 2.0)],
                2: [("q2", 3.0), ("q1", 4.0)],
            },
        )
        self.assertEqual(result.seed_count, 2)
        self.assertEqual(result.per_seed[1].mean_difference, 1.5)
        self.assertEqual(result.per_seed[2].mean_difference, 0.5)
        self.assertEqual(result.mean_difference, 1.0)
        self.assertAlmostEqual(result.difference_std, math.sqrt(0.5))

    def test_seed_summary_rejects_duplicate_and_missing_keys(self) -> None:
        with self.assertRaisesRegex(DuplicateKeyError, "duplicate key"):
            summarize_paired_seeds(
                [(1, "q1", 1.0), (1, "q1", 2.0)],
                [(1, "q1", 1.0)],
            )
        with self.assertRaisesRegex(MissingKeyError, "seed 1"):
            summarize_paired_seeds(
                {1: {"q1": 1.0, "q2": 2.0}},
                {1: {"q1": 1.0}},
            )
        with self.assertRaisesRegex(MissingKeyError, "seeds missing"):
            summarize_paired_seeds(
                {1: {"q1": 1.0}},
                {2: {"q1": 1.0}},
            )

    def test_non_finite_and_empty_observations_are_errors(self) -> None:
        with self.assertRaisesRegex(PairingError, "finite"):
            pair_by_key({"q": float("nan")}, {"q": 1.0})
        with self.assertRaisesRegex(PairingError, "at least one"):
            pair_by_key({}, {})


if __name__ == "__main__":
    unittest.main()
