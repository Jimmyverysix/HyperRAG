"""Core logic for auditing and filtering HyperRAG negative triples.

HyperRAG stores an n-ary hypergraph as an undirected incidence graph. Entity
nodes connect to hyperedge nodes, so one logical entity transition
``(head, hyperedge, tail)`` has length two in the stored graph. A transition
makes optimal progress toward an answer exactly when the tail is two incidence
edges closer to the nearest correct answer than the head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Iterable, Mapping, Sequence

import networkx as nx


Triplet = tuple[str, str, str]
NEGATIVE_POLICIES = ("original", "path-consistent-ignore")
HYPEREDGE_PREFIX = "<hyperedge>"
LOGICAL_TRANSITION_COST = 2


@dataclass(frozen=True)
class NegativeAudit:
    """Exact audit of the candidate negative pool for one query."""

    candidate_pool_size: int
    path_consistent_pool_size: int

    @property
    def path_consistent_pool_ratio(self) -> float:
        if self.candidate_pool_size == 0:
            return 0.0
        return self.path_consistent_pool_size / self.candidate_pool_size

    def to_dict(self) -> dict[str, int | float]:
        result = asdict(self)
        result["path_consistent_pool_ratio"] = self.path_consistent_pool_ratio
        return result


@dataclass(frozen=True)
class NegativeSamplingResult:
    """Negative samples plus the candidates masked by the selected policy."""

    negatives: list[Triplet]
    filtered: list[Triplet]
    originally_sampled_count: int
    path_consistent_sampled_count: int
    audit: NegativeAudit

    def to_dict(self) -> dict[str, int | float | str]:
        result: dict[str, int | float | str] = self.audit.to_dict()
        result.update(
            {
                "originally_sampled_count": self.originally_sampled_count,
                "path_consistent_sampled_count": self.path_consistent_sampled_count,
                "path_consistent_sampled_ratio": (
                    self.path_consistent_sampled_count / self.originally_sampled_count
                    if self.originally_sampled_count else 0.0
                ),
                "retained_sampled_count": len(self.negatives),
                "filtered_sampled_count": len(self.filtered),
            }
        )
        return result


def _is_hyperedge(node: object) -> bool:
    return str(node).startswith(HYPEREDGE_PREFIX)


def _normalize_triplets(triplets: Iterable[Sequence[str]]) -> set[Triplet]:
    return {tuple(triplet) for triplet in triplets}  # type: ignore[misc]


def _positive_lookup(positive_triplets: Iterable[Sequence[str]]) -> set[Triplet]:
    positives = _normalize_triplets(positive_triplets)
    return positives | {(tail, relation, head) for head, relation, tail in positives}


def answer_distances(
    graph: nx.Graph,
    answer_entities: Iterable[str],
) -> dict[str, int]:
    """Return unweighted distance from every reachable node to an answer set.

    For directed graphs the traversal is performed on the reverse graph because
    the desired quantity is distance *to* an answer. HyperRAG's incidence graphs
    are undirected, but the directed behavior also matches its KG representation.
    """

    answers = [answer for answer in dict.fromkeys(answer_entities) if answer in graph]
    if not answers:
        return {}

    distance_graph = graph.reverse(copy=False) if graph.is_directed() else graph
    return dict(
        nx.multi_source_dijkstra_path_length(
            distance_graph,
            answers,
            weight=None,
        )
    )


def is_path_consistent_transition(
    triplet: Sequence[str],
    distances: Mapping[str, int],
) -> bool:
    """Whether ``head -> hyperedge -> tail`` makes one optimal logical step."""

    head, _, tail = triplet
    head_distance = distances.get(head)
    tail_distance = distances.get(tail)
    if head_distance is None or tail_distance is None:
        return False
    return tail_distance == head_distance - LOGICAL_TRANSITION_COST


def filter_path_consistent_negatives(
    negative_triplets: Iterable[Sequence[str]],
    distances: Mapping[str, int],
) -> tuple[list[Triplet], list[Triplet]]:
    """Split negatives into retained and masked path-consistent candidates."""

    retained: list[Triplet] = []
    filtered: list[Triplet] = []
    for raw_triplet in negative_triplets:
        triplet = tuple(raw_triplet)  # type: ignore[assignment]
        target = filtered if is_path_consistent_transition(triplet, distances) else retained
        target.append(triplet)
    return retained, filtered


def candidate_negative_pool(
    subgraph: nx.Graph,
    positive_triplets: Iterable[Sequence[str]],
) -> set[Triplet]:
    """Enumerate the directed pseudo-binary triples eligible as negatives."""

    positives = _positive_lookup(positive_triplets)
    candidates: set[Triplet] = set()
    for hyperedge in subgraph.nodes:
        if not _is_hyperedge(hyperedge):
            continue
        entities = [
            str(node) for node in subgraph.neighbors(hyperedge) if not _is_hyperedge(node)
        ]
        for head in entities:
            for tail in entities:
                if head == tail:
                    continue
                triplet = (head, str(hyperedge), tail)
                if triplet not in positives:
                    candidates.add(triplet)
    return candidates


def audit_negative_pool(
    subgraph: nx.Graph,
    positive_triplets: Iterable[Sequence[str]],
    distances: Mapping[str, int],
) -> NegativeAudit:
    """Measure path-consistent candidates in the exact pre-sampling pool."""

    candidates = candidate_negative_pool(subgraph, positive_triplets)
    path_consistent_count = sum(
        is_path_consistent_transition(triplet, distances) for triplet in candidates
    )
    return NegativeAudit(
        candidate_pool_size=len(candidates),
        path_consistent_pool_size=path_consistent_count,
    )


def _sample_original_negatives(
    subgraph: nx.Graph,
    positive_triplets: Iterable[Sequence[str]],
    num_samples: int,
    rng=random,
) -> list[Triplet]:
    """Reproduce HyperRAG's released balanced random sampler."""

    if num_samples <= 0:
        return []

    positive_lookup = _positive_lookup(positive_triplets)
    hyperedges = [node for node in subgraph.nodes if _is_hyperedge(node)]
    if not hyperedges:
        return []

    negative_samples: set[Triplet] = set()
    max_attempts = num_samples * 20
    attempts = 0
    while len(negative_samples) < num_samples and attempts < max_attempts:
        attempts += 1
        hyperedge = rng.choice(hyperedges)
        connected_entities = [
            str(node) for node in subgraph.neighbors(hyperedge) if not _is_hyperedge(node)
        ]
        if len(connected_entities) < 2:
            continue
        head, tail = rng.sample(connected_entities, 2)
        triplet = (head, str(hyperedge), tail)
        if triplet not in positive_lookup and triplet not in negative_samples:
            negative_samples.add(triplet)
    return list(negative_samples)


def sample_negative_triplets(
    subgraph: nx.Graph,
    positive_triplets: Iterable[Sequence[str]],
    answer_entities: Iterable[str],
    num_samples: int,
    policy: str = "original",
    rng=random,
    distance_graph: nx.Graph | None = None,
) -> NegativeSamplingResult:
    """Sample HyperRAG negatives and optionally mask optimal-progress triples.

    The ignore policy first reproduces the original sampler and then removes
    path-consistent triples without replacement. This keeps every sampled
    negative shared with the baseline and changes only the disputed labels.
    """

    if policy not in NEGATIVE_POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; choose from {NEGATIVE_POLICIES}")

    positive_triplets = list(positive_triplets)
    graph_for_distance = distance_graph if distance_graph is not None else subgraph
    distances = answer_distances(graph_for_distance, answer_entities)
    audit = audit_negative_pool(subgraph, positive_triplets, distances)
    original = _sample_original_negatives(
        subgraph,
        positive_triplets,
        num_samples,
        rng,
    )

    clean_sampled, path_consistent_sampled = filter_path_consistent_negatives(
        original,
        distances,
    )
    if policy == "path-consistent-ignore":
        retained, filtered = clean_sampled, path_consistent_sampled
    else:
        retained, filtered = original, []

    return NegativeSamplingResult(
        negatives=retained,
        filtered=filtered,
        originally_sampled_count=len(original),
        path_consistent_sampled_count=len(path_consistent_sampled),
        audit=audit,
    )
