import unittest

import torch

from research.path_consistent_negative_learning.structured_data import (
    StructuredExperimentData,
    strategy_targets,
)


def toy_data() -> StructuredExperimentData:
    return StructuredExperimentData(
        domain="toy",
        sampler_seed=7,
        query_keys=["toy:0", "toy:1"],
        query_relations=torch.tensor([[0, 1, 2], [2, 1, 0]]),
        query_topics=torch.tensor([0, 4]),
        query_splits=torch.tensor([0, 2], dtype=torch.uint8),
        query_answer_ids=[(3,), (7,)],
        query_offsets=torch.tensor([0, 4, 8]),
        candidate_query_indices=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        candidate_transitions=torch.tensor(
            [[0, 0, 1], [1, 1, 3], [0, 2, 2], [2, 3, 3]] * 2
        ),
        numeric_features=torch.zeros((8, 9)),
        selected_labels=torch.tensor([1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.bool),
        disputed_labels=torch.tensor([0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.bool),
        hyperedge_relation_mask=torch.eye(8, 3),
        entity_count=8,
        relation_count=3,
    )


class StructuredStrategyTests(unittest.TestCase):
    def test_four_arms_share_shapes_and_match_mask_counts(self):
        data = toy_data()
        data.validate()
        assignments = {
            strategy: strategy_targets(data, strategy, seed=13)
            for strategy in (
                "strategy1_negative",
                "strategy2_ignore",
                "strategy3_positive",
                "random_drop",
            )
        }
        baseline_labels, baseline_mask = assignments["strategy1_negative"]
        ignore_labels, ignore_mask = assignments["strategy2_ignore"]
        positive_labels, positive_mask = assignments["strategy3_positive"]
        random_labels, random_mask = assignments["random_drop"]
        self.assertTrue(torch.equal(baseline_labels, ignore_labels))
        self.assertTrue(torch.equal(baseline_labels, random_labels))
        self.assertEqual(int((~ignore_mask).sum()), 2)
        self.assertEqual(int((~random_mask).sum()), 2)
        self.assertEqual(int(positive_labels.sum() - baseline_labels.sum()), 2)
        self.assertTrue(torch.all(baseline_mask))
        self.assertTrue(torch.all(positive_mask))

    def test_random_control_is_reproducible(self):
        data = toy_data()
        first = strategy_targets(data, "random_drop", seed=21)[1]
        second = strategy_targets(data, "random_drop", seed=21)[1]
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
