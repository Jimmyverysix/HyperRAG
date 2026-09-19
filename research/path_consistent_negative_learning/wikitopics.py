"""Deterministic WikiTopics KG loading and hypergraph audit primitives.

The public WikiTopics_QE graph is a binary KG.  For this audit, all outgoing
triples with the same head are grouped into one n-ary hyperedge.  The resulting
incidence graph is undirected and bipartite: one logical entity transition has
length two (entity -> hyperedge -> entity).

Pickle is intentionally used here because it is the public dataset format.
Only load dataset files obtained from a trusted source.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import pickle
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import networkx as nx


THREE_HOP_SHAPE = ("e", ("r", "r", "r"))
ENTITY_KIND = "entity"
HYPEREDGE_KIND = "hyperedge"
LOGICAL_TRANSITION_COST = 2
THREE_HOP_INCIDENCE_CUTOFF = 3 * LOGICAL_TRANSITION_COST

Node = tuple[str, int]
Transition = tuple[int, int, int]


def entity_node(entity_id: int) -> Node:
    """Return the collision-free incidence-graph node for an entity ID."""

    return (ENTITY_KIND, int(entity_id))


def hyperedge_node(owner_id: int) -> Node:
    """Return the hyperedge node owned by one KG head entity."""

    return (HYPEREDGE_KIND, int(owner_id))


def _node_sort_key(node: Node) -> tuple[int, int]:
    return (0 if node[0] == ENTITY_KIND else 1, node[1])


def _query_sort_key(query: "StructuredQuery") -> tuple[int, tuple[int, int, int]]:
    return (query.topic_id, query.relation_ids)


@dataclass(frozen=True, order=True)
class StructuredQuery:
    """The only query family in scope for this audit."""

    topic_id: int
    relation_ids: tuple[int, int, int]

    @classmethod
    def from_raw(cls, raw: object) -> "StructuredQuery":
        try:
            topic, relations = raw  # type: ignore[misc]
            relation_tuple = tuple(int(value) for value in relations)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed structured query: {raw!r}") from exc
        if len(relation_tuple) != 3:
            raise ValueError(f"Expected a three-hop query, got {raw!r}")
        return cls(int(topic), relation_tuple)  # type: ignore[arg-type]

    def as_raw(self) -> tuple[int, tuple[int, int, int]]:
        return (self.topic_id, self.relation_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "topic_id": self.topic_id,
            "relation_ids": list(self.relation_ids),
        }


@dataclass(frozen=True)
class QueryExample:
    query: StructuredQuery
    answer_ids: tuple[int, ...]


@dataclass(frozen=True)
class WikiTopicsDomain:
    """One loaded WikiTopics domain and its aggregated incidence graph."""

    name: str
    graph: nx.Graph
    examples: tuple[QueryExample, ...]
    mappings: Mapping[str, Any]
    triple_count: int
    unique_triple_count: int
    hyperedge_relations: Mapping[int, tuple[int, ...]]

    @property
    def entity_count(self) -> int:
        return sum(1 for node in self.graph if node[0] == ENTITY_KIND)

    @property
    def hyperedge_count(self) -> int:
        return sum(1 for node in self.graph if node[0] == HYPEREDGE_KIND)


@dataclass(frozen=True)
class QueryGraph:
    """Deterministic reconstruction of the released path-guided subgraph."""

    subgraph: nx.Graph
    selected_paths: tuple[tuple[Node, ...], ...]
    selected_positives: tuple[Transition, ...]
    reachable_answers: tuple[int, ...]
    missing_answers: tuple[int, ...]
    min_hop: int | None
    max_hop: int | None
    source_distances: Mapping[Node, int]


@dataclass(frozen=True)
class CandidatePoolProfile:
    """Exact candidate counts without materializing every ordered pair."""

    size: int
    by_arity: Mapping[int, int]
    by_depth: Mapping[int | None, int]


def _trusted_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - required by the dataset format


def _shape_payload(container: object, shape: object, label: str) -> object:
    if not isinstance(container, Mapping):
        raise ValueError(f"{label} pickle must contain a mapping")
    if shape in container:
        return container[shape]
    # Some mirrors serialize tuple-like keys slightly differently.  Equality
    # after recursive tuple normalization is sufficient and remains strict.
    normalized_shape = _deep_tuple(shape)
    for key, value in container.items():
        if _deep_tuple(key) == normalized_shape:
            return value
    raise ValueError(f"{label} pickle has no {shape!r} query family")


def _deep_tuple(value: object) -> object:
    if isinstance(value, (tuple, list)):
        return tuple(_deep_tuple(item) for item in value)
    return value


def load_query_examples(
    queries_path: Path,
    answers_path: Path,
    shape: object = THREE_HOP_SHAPE,
) -> tuple[QueryExample, ...]:
    """Load and join only ``('e', ('r', 'r', 'r'))`` train examples."""

    raw_queries = _shape_payload(_trusted_pickle(queries_path), shape, "queries")
    raw_answers = _shape_payload(_trusted_pickle(answers_path), shape, "answers")
    if not isinstance(raw_answers, Mapping):
        raise ValueError("answers payload must map structured queries to answer IDs")

    if isinstance(raw_queries, Mapping):
        query_values = raw_queries.keys()
    elif isinstance(raw_queries, Iterable) and not isinstance(raw_queries, (str, bytes)):
        query_values = raw_queries
    else:
        raise ValueError("queries payload must be an iterable of structured queries")

    answer_lookup = {_deep_tuple(key): value for key, value in raw_answers.items()}
    joined: list[QueryExample] = []
    seen: set[StructuredQuery] = set()
    for raw_query in query_values:
        query = StructuredQuery.from_raw(raw_query)
        if query in seen:
            continue
        seen.add(query)
        raw_answer_ids = answer_lookup.get(_deep_tuple(raw_query))
        if raw_answer_ids is None:
            raise ValueError(f"No hard answers for structured query {raw_query!r}")
        try:
            answer_ids = tuple(sorted({int(value) for value in raw_answer_ids}))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed answer IDs for query {raw_query!r}") from exc
        joined.append(QueryExample(query=query, answer_ids=answer_ids))

    return tuple(sorted(joined, key=lambda example: _query_sort_key(example.query)))


def load_aggregated_graph(
    graph_path: Path,
) -> tuple[nx.Graph, int, int, dict[int, tuple[int, ...]]]:
    """Group every head's outgoing KG edges into one n-ary hyperedge."""

    outgoing: dict[int, set[tuple[int, int]]] = defaultdict(set)
    triple_count = 0
    with graph_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(
                    f"{graph_path}:{line_number}: expected 'head relation tail'"
                )
            try:
                head_id, relation_id, tail_id = map(int, fields)
            except ValueError as exc:
                raise ValueError(
                    f"{graph_path}:{line_number}: graph IDs must be integers"
                ) from exc
            outgoing[head_id].add((relation_id, tail_id))
            triple_count += 1

    graph = nx.Graph()
    hyperedge_relations: dict[int, tuple[int, ...]] = {}
    for head_id in sorted(outgoing):
        facts = sorted(outgoing[head_id])
        incident_entities = sorted({head_id, *(tail for _, tail in facts)})
        relation_ids = tuple(sorted({relation for relation, _ in facts}))
        edge = hyperedge_node(head_id)
        graph.add_node(edge, kind=HYPEREDGE_KIND, owner_id=head_id, arity=len(incident_entities))
        hyperedge_relations[head_id] = relation_ids
        for entity_id in incident_entities:
            node = entity_node(entity_id)
            graph.add_node(node, kind=ENTITY_KIND, entity_id=entity_id)
            graph.add_edge(node, edge)

    unique_count = sum(len(facts) for facts in outgoing.values())
    return graph, triple_count, unique_count, hyperedge_relations


def load_domain(domain_dir: Path) -> WikiTopicsDomain:
    """Load the four public files required by the deterministic audit."""

    required = {
        "graph": domain_dir / "train_graph.txt",
        "queries": domain_dir / "train_queries.pkl",
        "answers": domain_dir / "train_answers_hard.pkl",
        "mappings": domain_dir / "og_mappings.pkl",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing WikiTopics files: " + ", ".join(missing))

    graph, triple_count, unique_count, hyperedge_relations = load_aggregated_graph(
        required["graph"]
    )
    examples = load_query_examples(required["queries"], required["answers"])
    mappings = _trusted_pickle(required["mappings"])
    if not isinstance(mappings, Mapping):
        raise ValueError("og_mappings.pkl must contain a mapping")
    return WikiTopicsDomain(
        name=domain_dir.name,
        graph=graph,
        examples=examples,
        mappings=mappings,
        triple_count=triple_count,
        unique_triple_count=unique_count,
        hyperedge_relations=hyperedge_relations,
    )


def shortest_distances(
    graph: nx.Graph,
    source: Node,
    cutoff: int | None = None,
) -> dict[Node, int]:
    if source not in graph:
        return {}
    return dict(nx.single_source_shortest_path_length(graph, source, cutoff=cutoff))


def _canonical_shortest_tree(
    graph: nx.Graph,
    source: Node,
    cutoff: int | None = None,
) -> tuple[dict[Node, Node | None], dict[Node, int]]:
    if source not in graph:
        return {}, {}
    parents: dict[Node, Node | None] = {source: None}
    distances: dict[Node, int] = {source: 0}
    queue: deque[Node] = deque([source])
    while queue:
        current = queue.popleft()
        if cutoff is not None and distances[current] >= cutoff:
            continue
        for neighbor in sorted(graph.neighbors(current), key=_node_sort_key):
            if neighbor not in parents:
                parents[neighbor] = current
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return parents, distances


def _path_from_tree(
    parents: Mapping[Node, Node | None], target: Node
) -> tuple[Node, ...] | None:
    if target not in parents:
        return None

    reversed_path: list[Node] = []
    node: Node | None = target
    while node is not None:
        reversed_path.append(node)
        node = parents[node]
    return tuple(reversed(reversed_path))


def canonical_shortest_path(graph: nx.Graph, source: Node, target: Node) -> tuple[Node, ...] | None:
    """Return a lexicographically tie-broken shortest path."""

    if source not in graph or target not in graph:
        return None
    parents, _ = _canonical_shortest_tree(graph, source)
    return _path_from_tree(parents, target)


def transitions_from_path(path: Sequence[Node]) -> tuple[Transition, ...]:
    transitions: list[Transition] = []
    for index in range(0, len(path) - 2, 2):
        head, edge, tail = path[index : index + 3]
        if head[0] != ENTITY_KIND or edge[0] != HYPEREDGE_KIND or tail[0] != ENTITY_KIND:
            raise ValueError(f"Non-alternating incidence path: {path!r}")
        transitions.append((head[1], edge[1], tail[1]))
    return tuple(transitions)


def build_query_graph(
    graph: nx.Graph,
    topic_id: int,
    answer_ids: Sequence[int],
) -> QueryGraph:
    """Reproduce HyperRAG's single-path positives and path-guided candidate graph."""

    source = entity_node(topic_id)
    # Every hard answer for this query family is generated by a three-relation
    # traversal in train_graph, hence it must be reachable within six incidence
    # edges.  The cutoff makes the full 11-domain audit local rather than doing
    # a whole connected-component traversal for every query.
    parents, source_distances = _canonical_shortest_tree(
        graph, source, cutoff=THREE_HOP_INCIDENCE_CUTOFF
    )
    selected: list[tuple[Node, ...]] = []
    reachable: list[int] = []
    missing: list[int] = []
    for answer_id in sorted(set(answer_ids)):
        path = _path_from_tree(parents, entity_node(answer_id))
        if path is None:
            missing.append(answer_id)
        else:
            selected.append(path)
            reachable.append(answer_id)

    if not selected:
        return QueryGraph(
            subgraph=nx.Graph(),
            selected_paths=(),
            selected_positives=(),
            reachable_answers=(),
            missing_answers=tuple(missing),
            min_hop=None,
            max_hop=None,
            source_distances=source_distances,
        )

    hop_counts = tuple((len(path) - 1) // LOGICAL_TRANSITION_COST for path in selected)
    selected_nodes = {node for path in selected for node in path}
    selected_positives = tuple(
        sorted({transition for path in selected for transition in transitions_from_path(path)})
    )

    subgraph_nodes: set[Node] = {source}
    subgraph_edges: set[tuple[Node, Node]] = set()
    to_expand: set[Node] = {source}
    expanded: set[Node] = set()
    for _ in range(max(hop_counts)):
        if not to_expand:
            break
        next_to_expand: set[Node] = set()
        expanded.update(to_expand)
        for entity in sorted(to_expand, key=_node_sort_key):
            for edge in sorted(graph.neighbors(entity), key=_node_sort_key):
                if edge[0] != HYPEREDGE_KIND:
                    continue
                subgraph_nodes.add(edge)
                subgraph_edges.add((entity, edge))
                for next_entity in sorted(graph.neighbors(edge), key=_node_sort_key):
                    if next_entity[0] != ENTITY_KIND:
                        continue
                    subgraph_nodes.add(next_entity)
                    subgraph_edges.add((edge, next_entity))
                    if next_entity in selected_nodes and next_entity not in expanded:
                        next_to_expand.add(next_entity)
        to_expand = next_to_expand

    subgraph = nx.Graph()
    for node in sorted(subgraph_nodes, key=_node_sort_key):
        subgraph.add_node(node, **graph.nodes[node])
    for first, second in sorted(
        subgraph_edges, key=lambda pair: (_node_sort_key(pair[0]), _node_sort_key(pair[1]))
    ):
        subgraph.add_edge(first, second)

    return QueryGraph(
        subgraph=subgraph,
        selected_paths=tuple(selected),
        selected_positives=selected_positives,
        reachable_answers=tuple(reachable),
        missing_answers=tuple(missing),
        min_hop=min(hop_counts),
        max_hop=max(hop_counts),
        source_distances=source_distances,
    )


def positive_lookup(positives: Iterable[Transition]) -> set[Transition]:
    direct = set(positives)
    return direct | {(tail, edge, head) for head, edge, tail in direct}


def candidate_pool(subgraph: nx.Graph, positives: Iterable[Transition]) -> tuple[Transition, ...]:
    """Enumerate the one shared directed candidate pool used by all policies."""

    excluded = positive_lookup(positives)
    candidates: set[Transition] = set()
    edges = sorted((node for node in subgraph if node[0] == HYPEREDGE_KIND), key=_node_sort_key)
    for edge in edges:
        entities = sorted(
            (node[1] for node in subgraph.neighbors(edge) if node[0] == ENTITY_KIND)
        )
        for head_id in entities:
            for tail_id in entities:
                if head_id == tail_id:
                    continue
                transition = (head_id, edge[1], tail_id)
                if transition not in excluded:
                    candidates.add(transition)
    return tuple(sorted(candidates))


def candidate_pool_profile(
    subgraph: nx.Graph,
    positives: Iterable[Transition],
    source_distances: Mapping[Node, int],
) -> CandidatePoolProfile:
    """Count the shared candidate pool in linear space.

    A hyperedge with ``k`` incident entities contributes ``k * (k - 1)``
    directed transitions before selected positives and their reverses are
    removed.  Counting by head also gives the exact depth strata.
    """

    excluded = positive_lookup(positives)
    by_arity: Counter[int] = Counter()
    by_depth: Counter[int | None] = Counter()
    total = 0
    edges = sorted(
        (node for node in subgraph if node[0] == HYPEREDGE_KIND),
        key=_node_sort_key,
    )
    for edge in edges:
        entity_ids = tuple(
            sorted(node[1] for node in subgraph.neighbors(edge) if node[0] == ENTITY_KIND)
        )
        entity_set = set(entity_ids)
        excluded_by_head: Counter[int] = Counter()
        for head_id, edge_id, tail_id in excluded:
            if (
                edge_id == edge[1]
                and head_id in entity_set
                and tail_id in entity_set
                and head_id != tail_id
            ):
                excluded_by_head[head_id] += 1

        arity = len(entity_ids)
        edge_count = arity * (arity - 1) - sum(excluded_by_head.values())
        total += edge_count
        by_arity[arity] += edge_count
        for head_id in entity_ids:
            head_count = arity - 1 - excluded_by_head[head_id]
            distance = source_distances.get(entity_node(head_id))
            depth = (
                distance // LOGICAL_TRANSITION_COST
                if distance is not None and distance % LOGICAL_TRANSITION_COST == 0
                else None
            )
            by_depth[depth] += head_count

    return CandidatePoolProfile(
        size=total,
        by_arity=dict(sorted(by_arity.items())),
        by_depth=dict(
            sorted(by_depth.items(), key=lambda item: (-1 if item[0] is None else item[0]))
        ),
    )


def _nodes_reaching_answers_on_shortest_paths(
    graph: nx.Graph,
    answer_ids: Sequence[int],
    source_distances: Mapping[Node, int],
) -> set[Node]:
    reachable_answers = {
        entity_node(answer_id)
        for answer_id in answer_ids
        if entity_node(answer_id) in source_distances
    }
    reaches_answer = set(reachable_answers)
    queue: deque[Node] = deque(sorted(reachable_answers, key=_node_sort_key))
    while queue:
        current = queue.popleft()
        current_distance = source_distances[current]
        for predecessor in graph.neighbors(current):
            if (
                source_distances.get(predecessor) == current_distance - 1
                and predecessor not in reaches_answer
            ):
                reaches_answer.add(predecessor)
                queue.append(predecessor)
    return reaches_answer


def disputed_transitions_in_pool(
    graph: nx.Graph,
    answer_ids: Sequence[int],
    subgraph: nx.Graph,
    positives: Iterable[Transition],
    source_distances: Mapping[Node, int],
) -> tuple[Transition, ...]:
    """Enumerate only unselected pool transitions in the shortest-path DAG."""

    reaches_answer = _nodes_reaching_answers_on_shortest_paths(
        graph, answer_ids, source_distances
    )
    excluded = positive_lookup(positives)
    disputed: set[Transition] = set()
    for edge in sorted(
        (node for node in subgraph if node[0] == HYPEREDGE_KIND),
        key=_node_sort_key,
    ):
        edge_distance = source_distances.get(edge)
        if edge_distance is None:
            continue
        entities = tuple(
            sorted(
                (node for node in subgraph.neighbors(edge) if node[0] == ENTITY_KIND),
                key=_node_sort_key,
            )
        )
        heads = (
            node for node in entities if source_distances.get(node) == edge_distance - 1
        )
        tails = tuple(
            node
            for node in entities
            if source_distances.get(node) == edge_distance + 1 and node in reaches_answer
        )
        for head in heads:
            for tail in tails:
                transition = (head[1], edge[1], tail[1])
                if transition not in excluded:
                    disputed.add(transition)
    return tuple(sorted(disputed))


def disputed_shortest_path_transitions(
    graph: nx.Graph,
    topic_id: int,
    answer_ids: Sequence[int],
    candidates: Iterable[Transition],
    source_distances: Mapping[Node, int] | None = None,
) -> tuple[Transition, ...]:
    """Find unselected candidates on any global topic--answer shortest path.

    This is pair-exact rather than merely testing distance to the nearest answer:
    a transition is retained when it lies on a shortest path from the topic to at
    least one reachable hard answer in the full incidence graph.
    """

    if source_distances is None:
        source_distances = shortest_distances(graph, entity_node(topic_id))

    # A node belongs to at least one source--answer shortest path exactly when
    # an answer is reachable from it through edges whose source distance rises
    # by one at every step.  Reverse traversal of that shortest-path DAG finds
    # the union for all answers in one pass, instead of one BFS per answer.
    reaches_answer = _nodes_reaching_answers_on_shortest_paths(
        graph, answer_ids, source_distances
    )

    disputed: list[Transition] = []
    for transition in sorted(set(candidates)):
        head_id, edge_id, tail_id = transition
        head = entity_node(head_id)
        edge = hyperedge_node(edge_id)
        tail = entity_node(tail_id)
        head_distance = source_distances.get(head)
        if head_distance is None:
            continue
        if (
            source_distances.get(edge) == head_distance + 1
            and source_distances.get(tail) == head_distance + LOGICAL_TRANSITION_COST
            and tail in reaches_answer
        ):
            disputed.append(transition)
    return tuple(disputed)


def simulate_original_sampler(
    subgraph: nx.Graph,
    positives: Iterable[Transition],
    num_samples: int,
    seed: int,
) -> tuple[Transition, ...]:
    """Deterministically simulate the released balanced random sampler."""

    if num_samples <= 0:
        return ()
    excluded = positive_lookup(positives)
    edges = sorted((node for node in subgraph if node[0] == HYPEREDGE_KIND), key=_node_sort_key)
    if not edges:
        return ()
    neighbors = {
        edge: tuple(
            sorted(node[1] for node in subgraph.neighbors(edge) if node[0] == ENTITY_KIND)
        )
        for edge in edges
    }
    rng = random.Random(seed)
    samples: set[Transition] = set()
    attempts = 0
    max_attempts = num_samples * 20
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        edge = rng.choice(edges)
        entities = neighbors[edge]
        if len(entities) < 2:
            continue
        head_id, tail_id = rng.sample(entities, 2)
        transition = (head_id, edge[1], tail_id)
        if transition not in excluded:
            samples.add(transition)
    return tuple(sorted(samples))


def transition_arity(graph: nx.Graph, transition: Transition) -> int:
    return int(graph.degree[hyperedge_node(transition[1])])


def transition_depth(
    source_distances: Mapping[Node, int], transition: Transition
) -> int | None:
    distance = source_distances.get(entity_node(transition[0]))
    if distance is None or distance % LOGICAL_TRANSITION_COST:
        return None
    return distance // LOGICAL_TRANSITION_COST


def transition_dict(transition: Transition, graph: nx.Graph, depth: int | None) -> dict[str, int | None]:
    head_id, edge_owner_id, tail_id = transition
    return {
        "head_id": head_id,
        "hyperedge_owner_id": edge_owner_id,
        "tail_id": tail_id,
        "arity": transition_arity(graph, transition),
        "depth": depth,
    }


def iter_mapping_sizes(mappings: Mapping[str, Any]) -> Iterator[tuple[str, int]]:
    """Yield stable, non-sensitive mapping cardinalities for provenance."""

    for raw_key, value in sorted(mappings.items(), key=lambda item: str(item[0])):
        if hasattr(value, "__len__"):
            yield str(raw_key), len(value)
