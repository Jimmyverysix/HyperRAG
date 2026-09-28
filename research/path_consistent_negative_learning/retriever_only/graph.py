"""Deterministic WikiTopics hypergraph reconstruction without an LLM."""

from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

import networkx as nx

from .data import DomainMappings, LabelSnapshot
from .labels import relation_base_id


HYPEREDGE_PREFIX = "H:"
LOGICAL_TRANSITION_COST = 2
MAX_INCIDENCE_DISTANCE = 6
Transition = tuple[str, str, str]


def hyperedge_id(owner_qid: str) -> str:
    return f"{HYPEREDGE_PREFIX}{owner_qid}"


def is_hyperedge(node: str) -> bool:
    return node.startswith(HYPEREDGE_PREFIX)


@dataclass(frozen=True)
class DeterministicHypergraph:
    graph: nx.Graph
    node_texts: Mapping[str, str]
    train_group_count: int
    test_group_count: int
    fact_count: int


def _decode_fact_groups(
    graph_path: Path,
    *,
    entities: Mapping[int, str],
    relations: Mapping[int, str],
    labels: LabelSnapshot,
) -> OrderedDict[str, tuple[tuple[str, str], ...]]:
    grouped: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
    with graph_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.strip().split()
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(f"{graph_path}:{line_number}: expected three integer IDs")
            try:
                head_id, relation_id, tail_id = map(int, fields)
            except ValueError as exc:
                raise ValueError(f"{graph_path}:{line_number}: non-integer ID") from exc
            head_qid = entities.get(head_id)
            tail_qid = entities.get(tail_id)
            relation_qid = relations.get(relation_id)
            if (
                head_qid not in labels.labels
                or tail_qid not in labels.labels
                or relation_qid is None
                or not labels.contains_relation(relation_qid)
            ):
                continue
            relation_text = labels.labels[relation_base_id(relation_qid)]
            if relation_qid.endswith("_inv"):
                relation_text += " (inverse)"
            # Keep Wikidata IDs as graph identity.  English labels are not
            # unique (for example, several entities are named "Union Station")
            # and therefore belong only in the encoder text.
            grouped.setdefault(head_qid, []).append((relation_text, tail_qid))

    return OrderedDict(
        (owner, tuple(dict.fromkeys(facts))) for owner, facts in grouped.items()
    )


def _fact_text(
    owner: str,
    facts: Sequence[tuple[str, str]],
    labels: LabelSnapshot,
) -> str:
    components = "; ".join(
        f"{relation}: {labels.labels[tail]}"
        for relation, tail in sorted(set(facts))
    )
    return f"{labels.labels[owner]} | {components}"


def build_deterministic_hypergraph(
    structured_domain_dir: Path,
    labels: LabelSnapshot,
) -> DeterministicHypergraph:
    """Build the train+test transductive hypergraph used by formal runs."""

    mappings = DomainMappings.load(structured_domain_dir / "og_mappings.pkl")
    train_groups = _decode_fact_groups(
        structured_domain_dir / "train_graph.txt",
        entities=mappings.train_entities,
        relations=mappings.relations,
        labels=labels,
    )
    test_groups = _decode_fact_groups(
        structured_domain_dir / "test_inference.txt",
        entities=mappings.test_entities,
        relations=mappings.relations,
        labels=labels,
    )

    facts_by_owner: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for owner, facts in (*train_groups.items(), *test_groups.items()):
        facts_by_owner[owner].update(facts)

    graph = nx.Graph()
    node_texts: dict[str, str] = {}
    fact_count = 0
    for owner in sorted(facts_by_owner):
        edge = hyperedge_id(owner)
        facts = sorted(facts_by_owner[owner])
        incident_entities = sorted({owner, *(tail for _, tail in facts)})
        graph.add_node(edge, kind="hyperedge", owner_qid=owner)
        node_texts[edge] = _fact_text(owner, facts, labels)
        for entity in incident_entities:
            graph.add_node(entity, kind="entity")
            graph.add_edge(entity, edge)
            node_texts[entity] = labels.labels[entity]
        fact_count += len(facts)
    return DeterministicHypergraph(
        graph=graph,
        node_texts=dict(sorted(node_texts.items())),
        train_group_count=len(train_groups),
        test_group_count=len(test_groups),
        fact_count=fact_count,
    )


def _ordered_neighbors(
    graph: nx.Graph,
    node: str,
    rng: random.Random | None,
) -> list[str]:
    values = sorted(graph.neighbors(node))
    if rng is not None:
        rng.shuffle(values)
    return values


def shortest_path(
    graph: nx.Graph,
    source: str,
    target: str,
    *,
    variant_seed: int | None = None,
    cutoff: int = MAX_INCIDENCE_DISTANCE,
) -> tuple[str, ...] | None:
    """Return one stable shortest path, optionally using seeded tie-breaking."""

    parents = _shortest_path_tree(
        graph,
        source,
        variant_seed=variant_seed,
        target=target,
        cutoff=cutoff,
    )
    return _path_from_tree(parents, target)


def _shortest_path_tree(
    graph: nx.Graph,
    source: str,
    *,
    variant_seed: int | None,
    target: str | None,
    cutoff: int,
) -> dict[str, str | None]:
    if source not in graph or (target is not None and target not in graph):
        return {}
    rng = None
    if variant_seed is not None:
        rng = random.Random(f"{variant_seed}|{source}|{target}")
    parents: dict[str, str | None] = {source: None}
    distances = {source: 0}
    queue: deque[str] = deque([source])
    while queue:
        current = queue.popleft()
        if target is not None and current == target:
            break
        if distances[current] >= cutoff:
            continue
        for neighbor in _ordered_neighbors(graph, current, rng):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    return parents


def _path_from_tree(
    parents: Mapping[str, str | None],
    target: str,
) -> tuple[str, ...] | None:
    if target not in parents:
        return None
    path: list[str] = []
    node: str | None = target
    while node is not None:
        path.append(node)
        node = parents[node]
    return tuple(reversed(path))


def transitions_from_path(path: Sequence[str]) -> tuple[Transition, ...]:
    transitions: list[Transition] = []
    for index in range(0, len(path) - 2, 2):
        head, edge, tail = path[index : index + 3]
        if is_hyperedge(head) or not is_hyperedge(edge) or is_hyperedge(tail):
            raise ValueError(f"Non-alternating incidence path: {path!r}")
        transitions.append((head, edge, tail))
    return tuple(transitions)


def selected_paths(
    graph: nx.Graph,
    topic: str,
    answers: Iterable[str],
    *,
    variant_seed: int | None = None,
) -> tuple[tuple[str, ...], ...]:
    answer_values = sorted(set(answers))
    paths = []
    shared_parents = None
    if variant_seed is None:
        shared_parents = _shortest_path_tree(
            graph,
            topic,
            variant_seed=None,
            target=None,
            cutoff=MAX_INCIDENCE_DISTANCE,
        )
    for answer in answer_values:
        path = (
            _path_from_tree(shared_parents, answer)
            if shared_parents is not None
            else shortest_path(
                graph,
                topic,
                answer,
                variant_seed=variant_seed,
            )
        )
        if path is not None:
            paths.append(path)
    return tuple(paths)


def path_guided_subgraph(
    graph: nx.Graph,
    topics: Sequence[str],
    paths: Sequence[Sequence[str]],
) -> nx.Graph:
    """Replay the released path-guided expansion with stable iteration order."""

    if not paths:
        return nx.Graph()
    path_nodes = {node for path in paths for node in path}
    maximum_hops = max((len(path) - 1) // LOGICAL_TRANSITION_COST for path in paths)
    nodes = set(topics)
    edges: set[tuple[str, str]] = set()
    frontier = set(topics)
    expanded: set[str] = set()
    for _ in range(maximum_hops):
        if not frontier:
            break
        expanded.update(frontier)
        next_frontier: set[str] = set()
        for entity in sorted(frontier):
            for hyperedge in sorted(graph.neighbors(entity)):
                if not is_hyperedge(hyperedge):
                    continue
                nodes.add(hyperedge)
                edges.add((entity, hyperedge))
                for next_entity in sorted(graph.neighbors(hyperedge)):
                    if is_hyperedge(next_entity):
                        continue
                    nodes.add(next_entity)
                    edges.add((hyperedge, next_entity))
                    if next_entity in path_nodes and next_entity not in expanded:
                        next_frontier.add(next_entity)
        frontier = next_frontier
    subgraph = nx.Graph()
    subgraph.add_nodes_from(sorted(nodes))
    subgraph.add_edges_from(sorted(edges))
    return subgraph


def all_shortest_distances(
    graph: nx.Graph,
    source: str,
    *,
    cutoff: int = MAX_INCIDENCE_DISTANCE,
) -> dict[str, int]:
    if source not in graph:
        return {}
    return dict(nx.single_source_shortest_path_length(graph, source, cutoff=cutoff))


def has_multiple_shortest_answer_paths(
    graph: nx.Graph,
    source: str,
    answers: Iterable[str],
    *,
    cutoff: int = MAX_INCIDENCE_DISTANCE,
) -> bool:
    """Return whether any source--answer pair has two shortest paths.

    Path counts are capped at two because sensitivity eligibility only needs to
    distinguish a unique shortest path from a tie; no path enumeration is
    performed.
    """

    distances = all_shortest_distances(graph, source, cutoff=cutoff)
    if not distances:
        return False
    path_counts = {source: 1}
    ordered_nodes = sorted(distances, key=lambda node: (distances[node], node))
    for node in ordered_nodes:
        if node == source:
            continue
        distance = distances[node]
        path_counts[node] = min(
            2,
            sum(
                path_counts.get(predecessor, 0)
                for predecessor in graph.neighbors(node)
                if distances.get(predecessor) == distance - 1
            ),
        )
    return any(path_counts.get(answer, 0) >= 2 for answer in set(answers))


def is_path_consistent(
    graph: nx.Graph,
    transition: Transition,
    topic: str,
    answers: Iterable[str],
) -> bool:
    """Apply the complete topic-answer pair distance equality."""

    head, edge, tail = transition
    if not graph.has_edge(head, edge) or not graph.has_edge(edge, tail):
        return False
    source_distances = all_shortest_distances(graph, topic)
    head_distance = source_distances.get(head)
    if head_distance is None:
        return False
    for answer in answers:
        topic_answer = source_distances.get(answer)
        if topic_answer is None:
            continue
        try:
            tail_answer = nx.shortest_path_length(graph, tail, answer)
        except nx.NetworkXNoPath:
            continue
        if head_distance + LOGICAL_TRANSITION_COST + tail_answer == topic_answer:
            return True
    return False
