"""Deterministic training and inference candidates for Retriever-only runs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable, Iterator, Mapping, Sequence

import networkx as nx

from .data import AlignedQuery
from .graph import (
    LOGICAL_TRANSITION_COST,
    MAX_INCIDENCE_DISTANCE,
    Transition,
    all_shortest_distances,
    is_hyperedge,
    path_guided_subgraph,
    selected_paths,
    transitions_from_path,
)


@dataclass(frozen=True)
class TrainingQuery:
    """Fixed candidates and labels shared by all loss-weight methods."""

    key: str
    text: str
    topic: str
    answers: tuple[str, ...]
    positives: tuple[Transition, ...]
    negatives: tuple[Transition, ...]
    path_consistent_negatives: tuple[Transition, ...]
    maximum_hops: int

    @property
    def candidates(self) -> tuple[Transition, ...]:
        return self.positives + self.negatives


@dataclass(frozen=True)
class RetrievalCandidate:
    transition: Transition
    first_hop: int


def _positive_lookup(positives: Iterable[Transition]) -> set[Transition]:
    direct = set(positives)
    return direct | {(tail, edge, head) for head, edge, tail in direct}


def sample_negative_transitions(
    subgraph: nx.Graph,
    positives: Sequence[Transition],
    *,
    rng: random.Random,
    count: int,
    maximum_attempt_multiplier: int = 20,
) -> tuple[Transition, ...]:
    """Replay the released equal-count sampler with an explicit RNG."""

    if count <= 0:
        return ()
    excluded = _positive_lookup(positives)
    hyperedges = sorted(node for node in subgraph if is_hyperedge(node))
    if not hyperedges:
        return ()
    sampled: set[Transition] = set()
    attempts = 0
    maximum_attempts = count * maximum_attempt_multiplier
    while len(sampled) < count and attempts < maximum_attempts:
        attempts += 1
        edge = rng.choice(hyperedges)
        entities = sorted(node for node in subgraph.neighbors(edge) if not is_hyperedge(node))
        if len(entities) < 2:
            continue
        head, tail = rng.sample(entities, 2)
        transition = (head, edge, tail)
        if transition not in excluded and transition not in sampled:
            sampled.add(transition)
    return tuple(sorted(sampled))


def shortest_path_dag_nodes(
    graph: nx.Graph,
    topic: str,
    answers: Iterable[str],
) -> tuple[dict[str, int], set[str]]:
    """Return source distances and nodes lying on any shortest answer path."""

    distances = all_shortest_distances(
        graph,
        topic,
        cutoff=MAX_INCIDENCE_DISTANCE,
    )
    reachable_answers = sorted(
        answer for answer in set(answers) if answer in distances
    )
    on_answer_path = set(reachable_answers)
    queue: deque[str] = deque(reachable_answers)
    while queue:
        current = queue.popleft()
        current_distance = distances[current]
        for predecessor in sorted(graph.neighbors(current)):
            if (
                distances.get(predecessor) == current_distance - 1
                and predecessor not in on_answer_path
            ):
                on_answer_path.add(predecessor)
                queue.append(predecessor)
    return distances, on_answer_path


def path_consistent_transitions(
    graph: nx.Graph,
    topic: str,
    answers: Iterable[str],
    candidates: Iterable[Transition],
) -> tuple[Transition, ...]:
    """Select candidates satisfying the complete topic-answer equality."""

    distances, on_answer_path = shortest_path_dag_nodes(graph, topic, answers)
    selected = []
    for transition in candidates:
        head, edge, tail = transition
        if not graph.has_edge(head, edge) or not graph.has_edge(edge, tail):
            continue
        head_distance = distances.get(head)
        tail_distance = distances.get(tail)
        if (
            head_distance is not None
            and tail_distance == head_distance + LOGICAL_TRANSITION_COST
            and tail in on_answer_path
        ):
            selected.append(transition)
    return tuple(sorted(set(selected)))


def build_training_query(
    query: AlignedQuery,
    graph: nx.Graph,
    *,
    rng: random.Random,
    variant_seed: int | None = None,
    maximum_attempt_multiplier: int = 20,
) -> TrainingQuery | None:
    """Build one fixed official-style training candidate set."""

    if (
        not query.alignment_supported
        or query.topic_node is None
        or query.topic_node not in graph
    ):
        return None
    answers = tuple(sorted(answer for answer in query.answer_nodes if answer in graph))
    if not answers:
        return None
    paths = selected_paths(
        graph,
        query.topic_node,
        answers,
        variant_seed=variant_seed,
    )
    if not paths:
        return None
    positives = tuple(
        sorted({transition for path in paths for transition in transitions_from_path(path)})
    )
    maximum_hops = max((len(path) - 1) // LOGICAL_TRANSITION_COST for path in paths)
    subgraph = path_guided_subgraph(graph, [query.topic_node], paths)
    negatives = sample_negative_transitions(
        subgraph,
        positives,
        rng=rng,
        count=len(positives),
        maximum_attempt_multiplier=maximum_attempt_multiplier,
    )
    consistent = path_consistent_transitions(
        graph,
        query.topic_node,
        answers,
        negatives,
    )
    return TrainingQuery(
        key=query.key,
        text=query.text,
        topic=query.topic_node,
        answers=answers,
        positives=positives,
        negatives=negatives,
        path_consistent_negatives=consistent,
        maximum_hops=maximum_hops,
    )


def build_training_queries(
    queries: Sequence[AlignedQuery],
    graph: nx.Graph,
    *,
    seed: int,
    variant_seed: int | None = None,
    maximum_attempt_multiplier: int = 20,
) -> Iterator[TrainingQuery]:
    """Yield all eligible queries using one domain-level sampler stream."""

    rng = random.Random(seed)
    for query in queries:
        prepared = build_training_query(
            query,
            graph,
            rng=rng,
            variant_seed=variant_seed,
            maximum_attempt_multiplier=maximum_attempt_multiplier,
        )
        if prepared is not None:
            yield prepared


def enumerate_retrieval_candidates(
    graph: nx.Graph,
    topic: str,
    *,
    maximum_hops: int = 3,
) -> tuple[RetrievalCandidate, ...]:
    """Exhaustively expand answer-free candidates from the known topic."""

    if topic not in graph:
        return ()
    frontier = {topic}
    expanded: set[str] = set()
    seen_pairs: set[tuple[str, str, str]] = set()
    output: list[RetrievalCandidate] = []
    for hop in range(1, maximum_hops + 1):
        next_frontier: set[str] = set()
        for head in sorted(frontier):
            if head in expanded:
                continue
            for edge in sorted(graph.neighbors(head)):
                if not is_hyperedge(edge):
                    continue
                for tail in sorted(graph.neighbors(edge)):
                    if is_hyperedge(tail) or tail == head:
                        continue
                    undirected_key = (edge, *sorted((head, tail)))
                    if undirected_key not in seen_pairs:
                        seen_pairs.add(undirected_key)
                        output.append(
                            RetrievalCandidate((head, edge, tail), first_hop=hop)
                        )
                    if tail not in expanded:
                        next_frontier.add(tail)
        expanded.update(frontier)
        frontier = next_frontier
        if not frontier:
            break
    return tuple(output)


def semantic_beam_candidates(
    graph: nx.Graph,
    topic: str,
    node_scores: Mapping[str, float],
    *,
    beam_width: int = 10,
    maximum_hops: int = 3,
) -> tuple[RetrievalCandidate, ...]:
    """Build a fixed answer-free beam using only frozen GTE similarities.

    Per head, the same beam width limits incident hyperedges and their tails;
    the globally best transitions determine the next frontier.  No trained MLP
    score or answer entity is consulted during candidate generation.
    """

    if topic not in graph:
        return ()
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    frontier = {topic}
    expanded: set[str] = set()
    selected_pairs: set[tuple[str, str, str]] = set()
    output: list[RetrievalCandidate] = []
    for hop in range(1, maximum_hops + 1):
        local_pairs: set[tuple[str, str, str]] = set()
        scored: list[tuple[float, Transition]] = []
        for head in sorted(frontier):
            if head in expanded:
                continue
            edges = sorted(
                (edge for edge in graph.neighbors(head) if is_hyperedge(edge)),
                key=lambda edge: (-node_scores[edge], edge),
            )[:beam_width]
            for edge in edges:
                tails = sorted(
                    (
                        tail
                        for tail in graph.neighbors(edge)
                        if not is_hyperedge(tail) and tail != head
                    ),
                    key=lambda tail: (-node_scores[tail], tail),
                )[:beam_width]
                for tail in tails:
                    pair = (edge, *sorted((head, tail)))
                    if pair in selected_pairs or pair in local_pairs:
                        continue
                    local_pairs.add(pair)
                    score = (
                        node_scores[head] + node_scores[edge] + node_scores[tail]
                    ) / 3.0
                    scored.append((score, (head, edge, tail)))
        chosen = sorted(scored, key=lambda item: (-item[0], item[1]))[:beam_width]
        if not chosen:
            break
        frontier = set()
        for _, transition in chosen:
            head, edge, tail = transition
            selected_pairs.add((edge, *sorted((head, tail))))
            output.append(RetrievalCandidate(transition, first_hop=hop))
            frontier.add(tail)
        expanded.update(head for _, (head, _, _) in chosen)
    return tuple(output)
