from __future__ import annotations

from dataclasses import fields
import unittest

import networkx as nx
import torch

from research.path_consistent_negative_learning.retriever_only.data import AlignedQuery
from research.path_consistent_negative_learning.retriever_only.embeddings import (
    EmbeddingStore,
)
from research.path_consistent_negative_learning.retriever_only.graph import (
    DeterministicHypergraph,
)
from research.path_consistent_negative_learning.retriever_only.official import (
    OfficialDDE,
)
from research.path_consistent_negative_learning.retriever_only.preparation import (
    prepare_training_candidates,
    prepare_training_candidates_multi,
)
from research.path_consistent_negative_learning.retriever_only.prepared import (
    PreparedCandidates,
)


def _add_transition(graph: nx.Graph, head: str, edge: str, tail: str) -> None:
    graph.add_edge(head, edge)
    graph.add_edge(edge, tail)


class _FakeDDE:
    def encode(self, transitions, topic):
        del topic
        return torch.arange(len(transitions) * 30, dtype=torch.float32).reshape(
            len(transitions), 30
        )

    def encode_many(self, groups):
        return [self.encode(transitions, topic) for transitions, topic in groups]


class RetrieverBatchPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        graph = nx.Graph()
        _add_transition(graph, "s", "H:1", "b")
        _add_transition(graph, "b", "H:2", "a")
        _add_transition(graph, "s", "H:3", "c")
        _add_transition(graph, "c", "H:4", "a")
        for edge, extra in (("H:1", "x1"), ("H:2", "x2"), ("H:3", "x3"), ("H:4", "x4")):
            graph.add_edge(edge, extra)
        self.bundle = DeterministicHypergraph(
            graph=graph,
            node_texts={node: node for node in graph},
            train_group_count=4,
            test_group_count=0,
            fact_count=8,
        )
        self.query = AlignedQuery(
            key="toy:train:1:1,2,3",
            domain="toy",
            split="train",
            raw_topic_id=1,
            raw_relation_ids=(1, 2, 3),
            topic_qid="s",
            relation_qids=("r1", "r2", "r3"),
            answer_qids=("a",),
            topic_node="s",
            relation_texts=("r1", "r2", "r3"),
            answer_nodes=("a",),
            text="toy question",
            topic_text_alignment_evidence=True,
            hard_answer_alignment_evidence=True,
            alignment_supported=True,
        )
        nodes = tuple(sorted(graph))
        self.embeddings = EmbeddingStore(
            node_names=nodes,
            node_embeddings=torch.zeros((len(nodes), 1024)),
            query_texts=(self.query.text,),
            query_embeddings=torch.zeros((1, 1024)),
        )

    def test_multi_seed_preparation_matches_separate_runs_exactly(self) -> None:
        seeds = (42, 43)
        separate = {
            seed: prepare_training_candidates(
                "toy",
                (self.query,),
                self.bundle,
                self.embeddings,
                _FakeDDE(),
                seed=seed,
                variant_seed=2718,
                progress_every=0,
            )
            for seed in seeds
        }
        combined = prepare_training_candidates_multi(
            "toy",
            (self.query,),
            self.bundle,
            self.embeddings,
            _FakeDDE(),
            seeds=seeds,
            variant_seed=2718,
            progress_every=0,
            dde_batch_size=1,
        )
        for seed in seeds:
            for field in fields(PreparedCandidates):
                expected = getattr(separate[seed], field.name)
                actual = getattr(combined[seed], field.name)
                if isinstance(expected, torch.Tensor):
                    self.assertTrue(torch.equal(actual, expected), field.name)
                else:
                    self.assertEqual(actual, expected, field.name)

    def test_grouped_official_dde_matches_individual_calls_exactly(self) -> None:
        groups = (
            (("s", "H:1", "b"), ("b", "H:2", "a")),
            (("x", "H:3", "y"), ("x", "H:4", "z")),
        )
        topics = ("s", "x")
        encoder = OfficialDDE(device="cpu")
        expected = [
            encoder.encode(transitions, topic)
            for transitions, topic in zip(groups, topics, strict=True)
        ]
        actual = encoder.encode_many(list(zip(groups, topics, strict=True)))
        for first, second in zip(expected, actual, strict=True):
            self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
