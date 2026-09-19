import unittest

import torch

from research.path_consistent_negative_learning.structured_model import (
    StructuredRetriever,
)


class StructuredRetrieverTests(unittest.TestCase):
    def test_forward_returns_one_logit_per_candidate(self):
        model = StructuredRetriever(
            entity_count=8,
            relation_count=4,
            hyperedge_relation_mask=torch.eye(8, 4),
            numeric_feature_count=9,
            embedding_dim=4,
            hidden_dim=16,
        )
        logits = model(
            entity_ids=torch.tensor([[0, 0, 1, 2], [3, 4, 5, 6]]),
            query_relation_ids=torch.tensor([[0, 1, 2], [1, 2, 3]]),
            numeric_features=torch.zeros((2, 9)),
        )
        self.assertEqual(logits.shape, (2,))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
