import unittest

import networkx as nx

from research.path_consistent_negative_learning.distances import (
    build_pair_distance_index,
    distances_to_any,
    makes_local_answer_progress,
    on_any_topic_answer_shortest_path,
)
from research.path_consistent_negative_learning.strategies import (
    apply_strategy,
    build_fixed_candidate_batch,
    random_drop,
    random_weighted,
    strategy1_negative,
    strategy2_ignore,
    strategy2_weighted,
    strategy3_positive,
)


def multi_pair_hypergraph() -> nx.Graph:
    """Two connected pairs, one disconnected pair, and one ternary fact."""

    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("S1", "<hyperedge>F1"),
            ("<hyperedge>F1", "X"),
            ("X", "<hyperedge>F2"),
            ("<hyperedge>F2", "A1"),
            ("S1", "<hyperedge>G1"),
            ("<hyperedge>G1", "Y"),
            ("<hyperedge>G1", "EXTRA"),
            ("Y", "<hyperedge>G2"),
            ("<hyperedge>G2", "A1"),
            ("S2", "<hyperedge>H1"),
            ("<hyperedge>H1", "Z"),
            ("Z", "<hyperedge>H2"),
            ("<hyperedge>H2", "A2"),
            ("S3", "<hyperedge>ISLAND"),
            ("A3", "<hyperedge>OTHER_ISLAND"),
        ]
    )
    return graph


class PairShortestPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = multi_pair_hypergraph()
        self.pairs = (("S1", "A1"), ("S2", "A2"), ("S3", "A3"))
        self.index = build_pair_distance_index(self.graph, self.pairs)

    def test_multiple_topics_and_answers_use_any_explicit_pair(self) -> None:
        self.assertTrue(
            on_any_topic_answer_shortest_path(
                self.graph,
                ("S1", "<hyperedge>G1", "Y"),
                self.index,
            )
        )
        self.assertTrue(
            on_any_topic_answer_shortest_path(
                self.graph,
                ("S2", "<hyperedge>H1", "Z"),
                self.index,
            )
        )

    def test_disconnected_pair_never_produces_false_membership(self) -> None:
        self.assertIsNone(self.index[2].topic_to_answer)
        self.assertFalse(
            on_any_topic_answer_shortest_path(
                self.graph,
                ("S3", "<hyperedge>ISLAND", "<dead-end>"),
                self.index,
            )
        )

    def test_reverse_transition_is_not_on_the_forward_shortest_path(self) -> None:
        self.assertFalse(
            on_any_topic_answer_shortest_path(
                self.graph,
                ("Y", "<hyperedge>G1", "S1"),
                self.index,
            )
        )

    def test_directed_graph_requires_the_transition_direction_to_exist(self) -> None:
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("S", "<hyperedge>F"),
                ("<hyperedge>F", "T"),
                ("T", "<hyperedge>G"),
                ("<hyperedge>G", "A"),
            ]
        )
        index = build_pair_distance_index(graph, (("S", "A"),))
        self.assertTrue(
            on_any_topic_answer_shortest_path(
                graph,
                ("S", "<hyperedge>F", "T"),
                index,
            )
        )
        self.assertFalse(
            on_any_topic_answer_shortest_path(
                graph,
                ("T", "<hyperedge>F", "S"),
                index,
            )
        )

    def test_high_arity_fact_checks_the_requested_head_and_tail(self) -> None:
        self.assertEqual(self.graph.degree["<hyperedge>G1"], 3)
        self.assertTrue(
            on_any_topic_answer_shortest_path(
                self.graph,
                ("S1", "<hyperedge>G1", "Y"),
                self.index,
            )
        )
        self.assertFalse(
            on_any_topic_answer_shortest_path(
                self.graph,
                ("S1", "<hyperedge>G1", "EXTRA"),
                self.index,
            )
        )

    def test_local_progress_is_retained_as_a_distinct_ablation_flag(self) -> None:
        transition = ("EXTRA", "<hyperedge>G1", "Y")
        answer_distances = distances_to_any(self.graph, ("A1", "A2", "A3"))
        self.assertTrue(
            makes_local_answer_progress(
                self.graph,
                transition,
                answer_distances,
            )
        )
        self.assertFalse(
            on_any_topic_answer_shortest_path(self.graph, transition, self.index)
        )


class StrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = multi_pair_hypergraph()
        self.transitions = (
            ("S1", "<hyperedge>F1", "X"),
            ("S1", "<hyperedge>G1", "Y"),
            ("Y", "<hyperedge>G1", "S1"),
            ("EXTRA", "<hyperedge>G1", "Y"),
            ("S1", "<hyperedge>G1", "EXTRA"),
        )
        self.features = tuple(object() for _ in self.transitions)
        self.batch = build_fixed_candidate_batch(
            graph=self.graph,
            query_id="question-1",
            transitions=self.transitions,
            selected_positive_transitions=(self.transitions[0],),
            topic_answer_pairs=(("S1", "A1"), ("S2", "A2")),
            features=self.features,
        )

    def test_four_arms_only_change_labels_and_loss_masks(self) -> None:
        assignments = (
            strategy1_negative(self.batch),
            strategy2_ignore(self.batch),
            strategy3_positive(self.batch),
            random_drop(self.batch, seed=7),
        )
        for assignment in assignments:
            self.assertIs(assignment.batch, self.batch)
            self.assertIs(assignment.batch.candidates, self.batch.candidates)
            for candidate, feature in zip(
                assignment.batch.candidates,
                self.features,
                strict=True,
            ):
                self.assertIs(candidate.features, feature)

        self.assertEqual(assignments[0].labels, (1, 0, 0, 0, 0))
        self.assertEqual(assignments[0].loss_mask, (True,) * 5)
        self.assertEqual(assignments[1].labels, assignments[0].labels)
        self.assertEqual(
            assignments[1].loss_mask,
            (True, False, True, True, True),
        )
        self.assertEqual(assignments[2].labels, (1, 1, 0, 0, 0))
        self.assertEqual(assignments[2].loss_mask, (True,) * 5)

    def test_local_ablation_uses_the_stored_local_flag(self) -> None:
        main = strategy2_ignore(self.batch)
        local = strategy2_ignore(
            self.batch,
            criterion="local_answer_progress",
        )
        self.assertEqual(main.ignored_count, 1)
        self.assertEqual(local.ignored_count, 2)
        self.assertFalse(local.loss_mask[3])

    def test_random_drop_matches_strategy2_count_for_each_question(self) -> None:
        batches = (
            self.batch,
            build_fixed_candidate_batch(
                graph=self.graph,
                query_id="question-2",
                transitions=self.transitions,
                selected_positive_transitions=(self.transitions[0],),
                topic_answer_pairs=(("S1", "A1"),),
                features=self.features,
            ),
        )
        for batch in batches:
            ignored = strategy2_ignore(batch)
            control = random_drop(batch, seed=19)
            self.assertEqual(control.ignored_count, ignored.ignored_count)
            self.assertTrue(control.loss_mask[0])
            self.assertEqual(control.labels, strategy1_negative(batch).labels)

    def test_random_drop_is_reproducible_and_varies_across_seeds(self) -> None:
        first = random_drop(self.batch, seed=23)
        repeated = random_drop(self.batch, seed=23)
        self.assertEqual(first.loss_mask, repeated.loss_mask)

        masks = {random_drop(self.batch, seed=seed).loss_mask for seed in range(20)}
        self.assertGreater(len(masks), 1)
        self.assertTrue(all(sum(not keep for keep in mask) == 1 for mask in masks))

    def test_named_dispatch_requires_a_seed_only_for_random_control(self) -> None:
        assignment = apply_strategy(self.batch, "strategy3_positive")
        self.assertEqual(assignment.labels, (1, 1, 0, 0, 0))
        with self.assertRaisesRegex(ValueError, "explicit seed"):
            apply_strategy(self.batch, "random_drop")

    def test_weighted_endpoints_match_strategies_1_and_2(self) -> None:
        ignored = strategy2_ignore(self.batch)
        weight_zero = strategy2_weighted(
            self.batch,
            disputed_negative_weight=0.0,
        )
        baseline = strategy1_negative(self.batch)
        weight_one = strategy2_weighted(
            self.batch,
            disputed_negative_weight=1.0,
        )
        self.assertEqual(weight_zero.labels, ignored.labels)
        self.assertEqual(weight_zero.loss_mask, ignored.loss_mask)
        self.assertEqual(weight_zero.loss_weights, ignored.loss_weights)
        self.assertEqual(weight_one.labels, baseline.labels)
        self.assertEqual(weight_one.loss_weights, baseline.loss_weights)

    def test_random_weighted_matches_count_without_touching_positives(self) -> None:
        assignment = random_weighted(
            self.batch,
            seed=17,
            disputed_negative_weight=0.25,
        )
        self.assertEqual(assignment.loss_weights[0], 1.0)
        self.assertEqual(assignment.loss_weights.count(0.25), 1)
        self.assertAlmostEqual(assignment.effective_weight, 4.25)


if __name__ == "__main__":
    unittest.main()
