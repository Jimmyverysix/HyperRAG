from __future__ import annotations

import json
from pathlib import Path
import pickle
import tempfile
import unittest

from research.path_consistent_negative_learning.retriever_only.data import (
    LabelSnapshot,
    load_aligned_queries,
)
from research.path_consistent_negative_learning.retriever_only.graph import (
    build_deterministic_hypergraph,
)


SHAPE = ("e", ("r", "r", "r"))
NLG_SHAPE = "('e', ('r', 'r', 'r'))"


class RetrieverOnlyAlignmentTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path, LabelSnapshot]:
        structured = root / "toy"
        nlg = root / "toy_nlg"
        structured.mkdir()
        nlg.mkdir()
        mappings = {
            "e2id_train": {"Q1": 1, "Q2": 2, "Q3": 3, "QX": 4},
            "e2id_test": {"Q1": 1, "Q2": 2, "Q3": 3, "QX": 4},
            "r2id": {"P1": 1, "P2": 2, "P3": 3},
        }
        with (structured / "og_mappings.pkl").open("wb") as handle:
            pickle.dump(mappings, handle)
        retained = (1, (1, 2, 3))
        filtered = (4, (1, 2, 3))
        for split in ("train", "valid", "test"):
            with (structured / f"{split}_queries.pkl").open("wb") as handle:
                pickle.dump({SHAPE: [retained, filtered]}, handle)
            with (structured / f"{split}_answers_hard.pkl").open("wb") as handle:
                pickle.dump({SHAPE: {retained: {3}, filtered: {2}}}, handle)
            (nlg / f"{split}_queries.json").write_text(
                json.dumps({NLG_SHAPE: ["Where is One?"]}),
                encoding="utf-8",
            )
            (nlg / f"{split}_answers_hard.json").write_text(
                json.dumps({NLG_SHAPE: {"Where is One?": ["Three"]}}),
                encoding="utf-8",
            )
        (structured / "train_graph.txt").write_text(
            "1 1 2\n1 2 3\n4 1 2\n",
            encoding="utf-8",
        )
        (structured / "test_inference.txt").write_text(
            "2 3 3\n",
            encoding="utf-8",
        )
        (nlg / "train_sentences.txt").write_text("Q1 links Q2 and Q3.\n", encoding="utf-8")
        (nlg / "test_sentences.txt").write_text("Q2 links Q3.\n", encoding="utf-8")
        labels = LabelSnapshot(
            labels={
                "Q1": "One",
                "Q2": "Two",
                "Q3": "Three",
                "P1": "first",
                "P2": "second",
                "P3": "third",
            },
            source="fixture",
            retrieved_at="2026-09-28T00:00:00+00:00",
        )
        return structured, nlg, labels

    def test_alignment_replays_missing_label_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            structured, nlg, labels = self._write_fixture(Path(temporary))
            queries = load_aligned_queries(structured, nlg, "train", labels)
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0].topic_qid, "Q1")
        self.assertEqual(queries[0].answer_qids, ("Q3",))
        self.assertEqual(queries[0].topic_node, "Q1")
        self.assertEqual(queries[0].answer_nodes, ("Q3",))
        self.assertEqual(queries[0].text, "Where is One?")
        self.assertTrue(queries[0].topic_text_alignment_evidence)
        self.assertTrue(queries[0].hard_answer_alignment_evidence)
        self.assertTrue(queries[0].alignment_supported)

    def test_graph_is_deterministically_serialized_after_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            structured, nlg, labels = self._write_fixture(Path(temporary))
            bundle = build_deterministic_hypergraph(structured, labels)
        self.assertEqual(bundle.train_group_count, 1)
        self.assertEqual(bundle.test_group_count, 1)
        self.assertEqual(bundle.fact_count, 3)
        self.assertIn("first: Two", bundle.node_texts["H:Q1"])
        self.assertTrue(bundle.graph.has_edge("Q1", "H:Q1"))
        self.assertFalse(bundle.graph.has_node("QX"))

    def test_equal_labels_do_not_merge_distinct_entities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            structured, _, labels = self._write_fixture(Path(temporary))
            collision_labels = LabelSnapshot(
                labels={**labels.labels, "QX": "One"},
                source=labels.source,
                retrieved_at=labels.retrieved_at,
            )
            bundle = build_deterministic_hypergraph(structured, collision_labels)
        self.assertNotEqual("Q1", "QX")
        self.assertEqual(bundle.node_texts["Q1"], bundle.node_texts["QX"])
        self.assertTrue(bundle.graph.has_edge("Q1", "H:Q1"))
        self.assertTrue(bundle.graph.has_edge("QX", "H:QX"))

    def test_alignment_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            structured, nlg, labels = self._write_fixture(Path(temporary))
            (nlg / "train_queries.json").write_text(
                json.dumps({NLG_SHAPE: ["first", "second", "unexpected"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "more queries"):
                load_aligned_queries(structured, nlg, "train", labels)


if __name__ == "__main__":
    unittest.main()
