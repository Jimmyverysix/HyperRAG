from __future__ import annotations

import unittest

import torch

from research.path_consistent_negative_learning.retriever_only.evaluation import (
    evaluate_scores,
)
from research.path_consistent_negative_learning.retriever_only.prepared import (
    PreparedCandidates,
)


class RetrieverEvaluationSubsetTests(unittest.TestCase):
    def test_sensitivity_uses_only_multiple_shortest_path_queries(self) -> None:
        data = PreparedCandidates(
            domain="toy",
            split="test",
            seed=None,
            query_keys=["q1", "q2"],
            query_topics=["s1", "s2"],
            query_answers=[("a1",), ("a2",)],
            query_offsets=torch.tensor([0, 1, 2]),
            query_embedding_indices=torch.zeros(2, dtype=torch.long),
            head_embedding_indices=torch.zeros(2, dtype=torch.long),
            edge_embedding_indices=torch.zeros(2, dtype=torch.long),
            tail_embedding_indices=torch.zeros(2, dtype=torch.long),
            dde_features=torch.zeros((2, 30)),
            transitions=[("s1", "H:1", "a1"), ("s2", "H:2", "a2")],
            query_multiple_shortest=[True, False],
        )
        result = evaluate_scores(
            data,
            torch.tensor([0.4, 0.3]),
            multiple_shortest_only=True,
        )
        self.assertEqual(result["evaluation_subset"], "multiple_equal_shortest_paths")
        self.assertEqual(result["query_count"], 1)
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual([row["query_key"] for row in result["queries"]], ["q1"])


if __name__ == "__main__":
    unittest.main()
