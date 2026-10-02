"""Reviewer snapshot of production source files; not a standalone module."""

# ===== BEGIN PRODUCTION SOURCE: research/path_consistent_negative_learning/retriever_only/candidates.py =====
"""Deterministic training and inference candidates for Retriever-only runs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Callable, Iterable, Iterator, Mapping, Sequence

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
class TrainingQueryStructure:
    """Seed-independent work shared by sensitivity training seeds."""

    key: str
    text: str
    topic: str
    answers: tuple[str, ...]
    positives: tuple[Transition, ...]
    negative_pool: nx.Graph
    source_distances: dict[str, int]
    shortest_path_nodes: set[str]
    maximum_hops: int


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
    *,
    neighbor_cache: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, int], set[str]]:
    """Return source distances and nodes lying on any shortest answer path."""

    distances = all_shortest_distances(
        graph,
        topic,
        cutoff=MAX_INCIDENCE_DISTANCE,
        neighbor_cache=neighbor_cache,
    )
    reachable_answers = sorted(
        answer for answer in set(answers) if answer in distances
    )
    on_answer_path = set(reachable_answers)
    queue: deque[str] = deque(reachable_answers)
    while queue:
        current = queue.popleft()
        current_distance = distances[current]
        predecessors = (
            neighbor_cache[current]
            if neighbor_cache is not None
            else sorted(graph.neighbors(current))
        )
        for predecessor in predecessors:
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
    return path_consistent_transitions_from_dag(
        graph,
        candidates,
        source_distances=distances,
        shortest_path_nodes=on_answer_path,
    )


def path_consistent_transitions_from_dag(
    graph: nx.Graph,
    candidates: Iterable[Transition],
    *,
    source_distances: dict[str, int],
    shortest_path_nodes: set[str],
) -> tuple[Transition, ...]:
    """Apply the path condition using a previously computed shortest-path DAG."""

    selected = []
    for transition in candidates:
        head, edge, tail = transition
        if not graph.has_edge(head, edge) or not graph.has_edge(edge, tail):
            continue
        head_distance = source_distances.get(head)
        tail_distance = source_distances.get(tail)
        if (
            head_distance is not None
            and tail_distance == head_distance + LOGICAL_TRANSITION_COST
            and tail in shortest_path_nodes
        ):
            selected.append(transition)
    return tuple(sorted(set(selected)))


def build_training_query_structure(
    query: AlignedQuery,
    graph: nx.Graph,
    *,
    variant_seed: int | None = None,
    neighbor_cache: Mapping[str, Sequence[str]] | None = None,
) -> TrainingQueryStructure | None:
    """Build the seed-independent portion of one training query."""

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
        neighbor_cache=neighbor_cache,
    )
    if not paths:
        return None
    positives = tuple(
        sorted({transition for path in paths for transition in transitions_from_path(path)})
    )
    maximum_hops = max((len(path) - 1) // LOGICAL_TRANSITION_COST for path in paths)
    negative_pool = path_guided_subgraph(
        graph,
        [query.topic_node],
        paths,
        neighbor_cache=neighbor_cache,
    )
    distances, shortest_path_nodes = shortest_path_dag_nodes(
        graph,
        query.topic_node,
        answers,
        neighbor_cache=neighbor_cache,
    )
    return TrainingQueryStructure(
        key=query.key,
        text=query.text,
        topic=query.topic_node,
        answers=answers,
        positives=positives,
        negative_pool=negative_pool,
        source_distances=distances,
        shortest_path_nodes=shortest_path_nodes,
        maximum_hops=maximum_hops,
    )


def sample_training_query(
    structure: TrainingQueryStructure,
    graph: nx.Graph,
    *,
    rng: random.Random,
    maximum_attempt_multiplier: int = 20,
) -> TrainingQuery:
    """Materialize the seed-dependent negatives for a cached query structure."""

    negatives = sample_negative_transitions(
        structure.negative_pool,
        structure.positives,
        rng=rng,
        count=len(structure.positives),
        maximum_attempt_multiplier=maximum_attempt_multiplier,
    )
    consistent = path_consistent_transitions_from_dag(
        graph,
        negatives,
        source_distances=structure.source_distances,
        shortest_path_nodes=structure.shortest_path_nodes,
    )
    return TrainingQuery(
        key=structure.key,
        text=structure.text,
        topic=structure.topic,
        answers=structure.answers,
        positives=structure.positives,
        negatives=negatives,
        path_consistent_negatives=consistent,
        maximum_hops=structure.maximum_hops,
    )


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
    node_score: Callable[[str], float],
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
                key=lambda edge: (-node_score(edge), edge),
            )[:beam_width]
            for edge in edges:
                tails = sorted(
                    (
                        tail
                        for tail in graph.neighbors(edge)
                        if not is_hyperedge(tail) and tail != head
                    ),
                    key=lambda tail: (-node_score(tail), tail),
                )[:beam_width]
                for tail in tails:
                    pair = (edge, *sorted((head, tail)))
                    if pair in selected_pairs or pair in local_pairs:
                        continue
                    local_pairs.add(pair)
                    score = (
                        node_score(head) + node_score(edge) + node_score(tail)
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
# ===== END PRODUCTION SOURCE: research/path_consistent_negative_learning/retriever_only/candidates.py =====

# ===== BEGIN PRODUCTION SOURCE: research/path_consistent_negative_learning/retriever_only/preparation.py =====
"""Prepare indexed GTE+DDE candidates without materializing 4126-D files."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Sequence

import torch

from .candidates import (
    TrainingQuery,
    build_training_query_structure,
    build_training_queries,
    sample_training_query,
    semantic_beam_candidates,
)
from .data import AlignedQuery
from .embeddings import EmbeddingStore
from .graph import (
    DeterministicHypergraph,
    Transition,
    has_multiple_shortest_answer_paths,
)
from .official import OfficialDDE
from .prepared import PreparedCandidates


@dataclass
class _Accumulator:
    domain: str
    split: str
    seed: int | None
    query_keys: list[str] = field(default_factory=list)
    query_topics: list[str] = field(default_factory=list)
    query_answers: list[tuple[str, ...]] = field(default_factory=list)
    query_offsets: list[int] = field(default_factory=lambda: [0])
    query_embedding_indices: list[int] = field(default_factory=list)
    head_embedding_indices: list[int] = field(default_factory=list)
    edge_embedding_indices: list[int] = field(default_factory=list)
    tail_embedding_indices: list[int] = field(default_factory=list)
    dde_features: list[torch.Tensor] = field(default_factory=list)
    labels: list[bool] | None = field(default_factory=list)
    path_consistent_mask: list[bool] | None = field(default_factory=list)
    transitions: list[Transition] | None = None
    query_multiple_shortest: list[bool] | None = None

    def add(
        self,
        *,
        key: str,
        text: str,
        topic: str,
        answers: tuple[str, ...],
        candidates: Sequence[Transition],
        dde: torch.Tensor,
        embeddings: EmbeddingStore,
        labels: Sequence[bool] | None,
        path_mask: Sequence[bool] | None,
        multiple_shortest: bool | None = None,
    ) -> None:
        node_index = embeddings.node_index
        query_index = embeddings.query_index
        self.query_keys.append(key)
        self.query_topics.append(topic)
        self.query_answers.append(answers)
        for head, edge, tail in candidates:
            self.query_embedding_indices.append(query_index[text])
            self.head_embedding_indices.append(node_index[head])
            self.edge_embedding_indices.append(node_index[edge])
            self.tail_embedding_indices.append(node_index[tail])
        self.dde_features.append(dde)
        if self.labels is not None and labels is not None:
            self.labels.extend(labels)
        if self.path_consistent_mask is not None and path_mask is not None:
            self.path_consistent_mask.extend(path_mask)
        if self.transitions is not None:
            self.transitions.extend(candidates)
        if self.query_multiple_shortest is not None:
            if multiple_shortest is None:
                raise ValueError("evaluation queries require a shortest-path flag")
            self.query_multiple_shortest.append(multiple_shortest)
        self.query_offsets.append(self.query_offsets[-1] + len(candidates))

    def finish(self) -> PreparedCandidates:
        dde = (
            torch.cat(self.dde_features, dim=0)
            if self.dde_features
            else torch.empty((0, 30), dtype=torch.float32)
        )
        value = PreparedCandidates(
            domain=self.domain,
            split=self.split,
            seed=self.seed,
            query_keys=self.query_keys,
            query_topics=self.query_topics,
            query_answers=self.query_answers,
            query_offsets=torch.tensor(self.query_offsets, dtype=torch.long),
            query_embedding_indices=torch.tensor(
                self.query_embedding_indices, dtype=torch.long
            ),
            head_embedding_indices=torch.tensor(
                self.head_embedding_indices, dtype=torch.long
            ),
            edge_embedding_indices=torch.tensor(
                self.edge_embedding_indices, dtype=torch.long
            ),
            tail_embedding_indices=torch.tensor(
                self.tail_embedding_indices, dtype=torch.long
            ),
            dde_features=dde,
            labels=(
                torch.tensor(self.labels, dtype=torch.bool)
                if self.labels is not None
                else None
            ),
            path_consistent_mask=(
                torch.tensor(self.path_consistent_mask, dtype=torch.bool)
                if self.path_consistent_mask is not None
                else None
            ),
            transitions=self.transitions,
            query_multiple_shortest=self.query_multiple_shortest,
        )
        value.validate()
        return value


def prepare_training_candidates(
    domain: str,
    queries: Sequence[AlignedQuery],
    bundle: DeterministicHypergraph,
    embeddings: EmbeddingStore,
    dde_encoder: OfficialDDE,
    *,
    seed: int,
    variant_seed: int | None = None,
    progress_every: int = 500,
) -> PreparedCandidates:
    accumulator = _Accumulator(domain=domain, split="train", seed=seed)
    for index, query in enumerate(
        build_training_queries(
            queries,
            bundle.graph,
            seed=seed,
            variant_seed=variant_seed,
        ),
        start=1,
    ):
        candidates = query.candidates
        path_consistent = set(query.path_consistent_negatives)
        accumulator.add(
            key=query.key,
            text=query.text,
            topic=query.topic,
            answers=query.answers,
            candidates=candidates,
            dde=dde_encoder.encode(candidates, query.topic),
            embeddings=embeddings,
            labels=[True] * len(query.positives) + [False] * len(query.negatives),
            path_mask=[False] * len(query.positives)
            + [value in path_consistent for value in query.negatives],
        )
        if progress_every and index % progress_every == 0:
            print(f"[{domain}/train/seed={seed}] {index} queries", flush=True)
    return accumulator.finish()


def prepare_training_candidates_multi(
    domain: str,
    queries: Sequence[AlignedQuery],
    bundle: DeterministicHypergraph,
    embeddings: EmbeddingStore,
    dde_encoder: OfficialDDE,
    *,
    seeds: Sequence[int],
    variant_seed: int | None = None,
    progress_every: int = 500,
) -> dict[int, PreparedCandidates]:
    """Prepare several seeds while sharing all seed-independent graph work.

    Each seed owns the same domain-level ``random.Random`` stream used by
    :func:`prepare_training_candidates`.  Only shortest paths, the guided
    subgraph, and the shortest-path DAG are shared, so every emitted tensor is
    semantically identical to preparing that seed in a separate process.
    """

    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or len(seed_values) != len(set(seed_values)):
        raise ValueError("seeds must be nonempty and unique")
    random_streams = {seed: random.Random(seed) for seed in seed_values}
    neighbor_cache = {
        node: tuple(sorted(bundle.graph.neighbors(node)))
        for node in bundle.graph
    }
    prepared_by_seed: dict[int, list[TrainingQuery]] = {
        seed: [] for seed in seed_values
    }
    prepared_query_count = 0
    for query in queries:
        structure = build_training_query_structure(
            query,
            bundle.graph,
            variant_seed=variant_seed,
            neighbor_cache=neighbor_cache,
        )
        if structure is None:
            continue
        prepared_query_count += 1
        for seed in seed_values:
            prepared = sample_training_query(
                structure,
                bundle.graph,
                rng=random_streams[seed],
            )
            prepared_by_seed[seed].append(prepared)
        if progress_every and prepared_query_count % progress_every == 0:
            seed_text = ",".join(str(seed) for seed in seed_values)
            print(
                f"[{domain}/train/seeds={seed_text}] {prepared_query_count} queries",
                flush=True,
            )

    output = {}
    for seed in seed_values:
        accumulator = _Accumulator(domain=domain, split="train", seed=seed)
        for prepared in prepared_by_seed[seed]:
            candidates = prepared.candidates
            path_consistent = set(prepared.path_consistent_negatives)
            accumulator.add(
                key=prepared.key,
                text=prepared.text,
                topic=prepared.topic,
                answers=prepared.answers,
                candidates=candidates,
                dde=dde_encoder.encode(candidates, prepared.topic),
                embeddings=embeddings,
                labels=[True] * len(prepared.positives)
                + [False] * len(prepared.negatives),
                path_mask=[False] * len(prepared.positives)
                + [value in path_consistent for value in prepared.negatives],
            )
        output[seed] = accumulator.finish()
    return output


def prepare_evaluation_candidates(
    domain: str,
    split: str,
    queries: Sequence[AlignedQuery],
    bundle: DeterministicHypergraph,
    embeddings: EmbeddingStore,
    dde_encoder: OfficialDDE,
    *,
    similarity_device: str,
    beam_width: int = 10,
    progress_every: int = 500,
) -> PreparedCandidates:
    accumulator = _Accumulator(
        domain=domain,
        split=split,
        seed=None,
        labels=None,
        path_consistent_mask=None,
        transitions=[],
        query_multiple_shortest=[],
    )
    device = torch.device(similarity_device)
    node_embeddings = embeddings.node_embeddings.to(device)
    node_index = embeddings.node_index
    query_index = embeddings.query_index
    eligible = 0
    for query in queries:
        if (
            not query.alignment_supported
            or query.topic_node is None
            or query.topic_node not in bundle.graph
        ):
            continue
        answers = tuple(
            sorted(answer for answer in query.answer_nodes if answer in bundle.graph)
        )
        if not answers:
            continue
        eligible += 1
        query_embedding = embeddings.query_embeddings[query_index[query.text]].to(device)
        with torch.no_grad():
            similarities = torch.mv(node_embeddings, query_embedding).cpu().tolist()
        def node_score(name: str) -> float:
            return similarities[node_index[name]]

        candidates = tuple(
            value.transition
            for value in semantic_beam_candidates(
                bundle.graph,
                query.topic_node,
                node_score,
                beam_width=beam_width,
            )
        )
        accumulator.add(
            key=query.key,
            text=query.text,
            topic=query.topic_node,
            answers=answers,
            candidates=candidates,
            dde=dde_encoder.encode(candidates, query.topic_node),
            embeddings=embeddings,
            labels=None,
            path_mask=None,
            multiple_shortest=has_multiple_shortest_answer_paths(
                bundle.graph,
                query.topic_node,
                answers,
            ),
        )
        if progress_every and eligible % progress_every == 0:
            print(f"[{domain}/{split}] {eligible} queries", flush=True)
    return accumulator.finish()
# ===== END PRODUCTION SOURCE: research/path_consistent_negative_learning/retriever_only/preparation.py =====
