"""Shortest-path indexes and transition predicates for incidence graphs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

import networkx as nx

from .schema import Node, TopicAnswerPair, Transition


LOGICAL_TRANSITION_COST = 2


@dataclass(frozen=True)
class PairDistances:
    """Distances needed to test shortest-path membership for one ``(s, a)``."""

    topic: Node
    answer: Node
    from_topic: Mapping[Node, int]
    to_answer: Mapping[Node, int]
    topic_to_answer: int | None


def _frozen_distances(distances: Mapping[Node, int]) -> Mapping[Node, int]:
    return MappingProxyType(dict(distances))


def distances_from(graph: nx.Graph, source: Node) -> Mapping[Node, int]:
    """Return unweighted directed distances from ``source``."""

    if source not in graph:
        return MappingProxyType({})
    return _frozen_distances(nx.single_source_shortest_path_length(graph, source))


def distances_to(graph: nx.Graph, target: Node) -> Mapping[Node, int]:
    """Return unweighted directed distances to ``target``."""

    if target not in graph:
        return MappingProxyType({})
    traversal_graph = graph.reverse(copy=False) if graph.is_directed() else graph
    return _frozen_distances(
        nx.single_source_shortest_path_length(traversal_graph, target)
    )


def distances_to_any(
    graph: nx.Graph,
    targets: Iterable[Node],
) -> Mapping[Node, int]:
    """Return the distance from every reachable node to its nearest target."""

    present_targets = tuple(dict.fromkeys(t for t in targets if t in graph))
    if not present_targets:
        return MappingProxyType({})
    traversal_graph = graph.reverse(copy=False) if graph.is_directed() else graph
    distances = nx.multi_source_dijkstra_path_length(
        traversal_graph,
        present_targets,
        weight=None,
    )
    return _frozen_distances(distances)


def build_pair_distance_index(
    graph: nx.Graph,
    topic_answer_pairs: Iterable[TopicAnswerPair],
) -> tuple[PairDistances, ...]:
    """Build a reusable distance index for explicit topic--answer pairs.

    Pairs are explicit rather than a Cartesian product.  This matters for
    questions with several topics and answers whose valid alignments are known.
    Duplicate pairs are removed without changing their first-seen order.
    """

    unique_pairs = tuple(dict.fromkeys(tuple(pair) for pair in topic_answer_pairs))
    from_cache: dict[Node, Mapping[Node, int]] = {}
    to_cache: dict[Node, Mapping[Node, int]] = {}
    entries: list[PairDistances] = []

    for topic, answer in unique_pairs:
        from_topic = from_cache.setdefault(topic, distances_from(graph, topic))
        to_answer = to_cache.setdefault(answer, distances_to(graph, answer))
        entries.append(
            PairDistances(
                topic=topic,
                answer=answer,
                from_topic=from_topic,
                to_answer=to_answer,
                topic_to_answer=from_topic.get(answer),
            )
        )
    return tuple(entries)


def is_valid_transition(graph: nx.Graph, transition: Transition) -> bool:
    """Whether ``head -> fact -> tail`` is present in the incidence graph."""

    head, fact, tail = transition
    return graph.has_edge(head, fact) and graph.has_edge(fact, tail)


def on_any_topic_answer_shortest_path(
    graph: nx.Graph,
    transition: Transition,
    pair_index: Iterable[PairDistances],
) -> bool:
    """Test the proposal's main ambiguity definition.

    A valid transition ``(h, f, t)`` is accepted iff at least one supplied
    ``(s, a)`` pair satisfies

    ``d(s, h) + 2 + d(t, a) == d(s, a)``.
    """

    if not is_valid_transition(graph, transition):
        return False

    head, _, tail = transition
    for entry in pair_index:
        source_to_head = entry.from_topic.get(head)
        tail_to_answer = entry.to_answer.get(tail)
        pair_distance = entry.topic_to_answer
        if (
            source_to_head is not None
            and tail_to_answer is not None
            and pair_distance is not None
            and source_to_head + LOGICAL_TRANSITION_COST + tail_to_answer
            == pair_distance
        ):
            return True
    return False


def makes_local_answer_progress(
    graph: nx.Graph,
    transition: Transition,
    answer_distance_index: Mapping[Node, int],
) -> bool:
    """Test the older, topic-agnostic nearest-answer rule for ablations."""

    if not is_valid_transition(graph, transition):
        return False
    head, _, tail = transition
    head_distance = answer_distance_index.get(head)
    tail_distance = answer_distance_index.get(tail)
    return (
        head_distance is not None
        and tail_distance is not None
        and tail_distance == head_distance - LOGICAL_TRANSITION_COST
    )
