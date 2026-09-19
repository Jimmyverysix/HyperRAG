import json
from pathlib import Path
import pickle
import tempfile
import unittest

from research.path_consistent_negative_learning.audit_wikitopics import (
    audit_domain_to_files,
    audit_example,
)
from research.path_consistent_negative_learning.wikitopics import (
    THREE_HOP_SHAPE,
    build_query_graph,
    candidate_pool,
    disputed_shortest_path_transitions,
    entity_node,
    hyperedge_node,
    load_domain,
    shortest_distances,
)


def write_fixture(root: Path) -> Path:
    """Create two equal two-hop routes: 0--1--2 and 0--3--2."""

    domain_dir = root / "toy"
    domain_dir.mkdir(parents=True)
    (domain_dir / "train_graph.txt").write_text(
        "0 10 1\n"
        "0 11 3\n"
        "1 12 2\n"
        "3 13 2\n"
        "8 14 9\n",
        encoding="utf-8",
    )
    query = (0, (10, 12, 99))
    with (domain_dir / "train_queries.pkl").open("wb") as handle:
        pickle.dump(
            {
                THREE_HOP_SHAPE: {query},
                ("e", ("r",)): {(8, (14,))},
            },
            handle,
        )
    with (domain_dir / "train_answers_hard.pkl").open("wb") as handle:
        pickle.dump(
            {
                THREE_HOP_SHAPE: {query: {2}},
                ("e", ("r",)): {(8, (14,)): {9}},
            },
            handle,
        )
    with (domain_dir / "og_mappings.pkl").open("wb") as handle:
        pickle.dump(
            {
                "e2id_train": {f"Q{value}": value for value in (0, 1, 2, 3, 8, 9)},
                "r2id": {f"P{value}": value for value in (10, 11, 12, 13, 14)},
            },
            handle,
        )
    return domain_dir


class WikiTopicsLoadingTests(unittest.TestCase):
    def test_groups_all_outgoing_facts_by_head_and_filters_query_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            domain = load_domain(write_fixture(Path(temporary)))

        self.assertEqual(domain.triple_count, 5)
        self.assertEqual(domain.hyperedge_count, 4)
        self.assertEqual(domain.graph.degree[hyperedge_node(0)], 3)
        self.assertEqual(domain.hyperedge_relations[0], (10, 11))
        self.assertEqual(len(domain.examples), 1)
        self.assertEqual(domain.examples[0].query.topic_id, 0)
        self.assertEqual(domain.examples[0].answer_ids, (2,))


class WikiTopicsAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.domain = load_domain(write_fixture(self.root))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_audit_finds_unselected_equal_shortest_path_branch(self) -> None:
        record = audit_example(
            self.domain,
            self.domain.examples[0],
            query_index=0,
            seeds=(7, 11),
        )

        selected = {
            (item["head_id"], item["hyperedge_owner_id"], item["tail_id"])
            for item in record["selected_positive_transitions"]
        }
        disputed = {
            (item["head_id"], item["hyperedge_owner_id"], item["tail_id"])
            for item in record["disputed_transitions"]
        }
        self.assertIn((0, 0, 1), selected)
        self.assertNotIn((0, 0, 3), selected)
        self.assertIn((0, 0, 3), disputed)
        self.assertGreater(record["candidate_pool_size"], 0)
        self.assertEqual(record["candidate_pool_policy"], "shared_across_label_strategies")
        self.assertEqual([item["seed"] for item in record["sampling"]], [7, 11])
        self.assertIn("hop", record["strata"])
        self.assertIn("arity", record["strata"])
        self.assertIn("depth", record["strata"])

    def test_shared_pool_excludes_selected_transition_and_reverse(self) -> None:
        query_graph = build_query_graph(self.domain.graph, 0, (2,))
        pool = set(candidate_pool(query_graph.subgraph, query_graph.selected_positives))
        self.assertNotIn((0, 0, 1), pool)
        self.assertNotIn((1, 0, 0), pool)
        self.assertIn((0, 0, 3), pool)

    def test_shortest_path_dag_matches_pairwise_distance_definition(self) -> None:
        query_graph = build_query_graph(self.domain.graph, 0, (2, 9))
        candidates = candidate_pool(
            query_graph.subgraph, query_graph.selected_positives
        )
        actual = set(
            disputed_shortest_path_transitions(
                self.domain.graph,
                0,
                (2, 9),
                candidates,
                source_distances=query_graph.source_distances,
            )
        )

        source_distances = query_graph.source_distances
        expected = set()
        for answer_id in (2, 9):
            answer = entity_node(answer_id)
            pair_distance = source_distances.get(answer)
            if pair_distance is None:
                continue
            target_distances = shortest_distances(
                self.domain.graph, answer, cutoff=pair_distance
            )
            for head_id, edge_id, tail_id in candidates:
                if (
                    source_distances.get(entity_node(head_id), 10**9)
                    + 2
                    + target_distances.get(entity_node(tail_id), 10**9)
                    == pair_distance
                ):
                    expected.add((head_id, edge_id, tail_id))
        self.assertEqual(actual, expected)

    def test_jsonl_and_summary_are_byte_reproducible(self) -> None:
        output = self.root / "output"
        jsonl_path = output / "toy.audit.jsonl"
        summary_path = output / "toy.summary.json"
        first = audit_domain_to_files(
            self.root / "toy", jsonl_path, summary_path, seeds=(11, 7)
        )
        first_jsonl = jsonl_path.read_bytes()
        first_summary = summary_path.read_bytes()
        second = audit_domain_to_files(
            self.root / "toy", jsonl_path, summary_path, seeds=(7, 11)
        )

        self.assertEqual(first, second)
        self.assertEqual(first_jsonl, jsonl_path.read_bytes())
        self.assertEqual(first_summary, summary_path.read_bytes())
        self.assertEqual(first["query_count"], 1)
        self.assertGreater(first["disputed_candidate_count"], 0)
        row = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
        self.assertEqual(row["query"]["topic_id"], 0)


if __name__ == "__main__":
    unittest.main()
