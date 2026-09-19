"""Fixed structured WikiTopics candidates for the Gate C proxy experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Mapping

import torch

from .strategies import STRATEGIES
from .wikitopics import (
    HYPEREDGE_KIND,
    LOGICAL_TRANSITION_COST,
    WikiTopicsDomain,
    build_query_graph,
    disputed_shortest_path_transitions,
    entity_node,
    hyperedge_node,
    load_domain,
    simulate_original_sampler,
)


TRAIN_SPLIT = 0
VALIDATION_SPLIT = 1
TEST_SPLIT = 2
SPLIT_NAMES = {TRAIN_SPLIT: "train", VALIDATION_SPLIT: "validation", TEST_SPLIT: "test"}


@dataclass
class StructuredExperimentData:
    """Tensorized candidates grouped contiguously by query."""

    domain: str
    sampler_seed: int
    query_keys: list[str]
    query_relations: torch.Tensor
    query_topics: torch.Tensor
    query_splits: torch.Tensor
    query_answer_ids: list[tuple[int, ...]]
    query_offsets: torch.Tensor
    candidate_query_indices: torch.Tensor
    candidate_transitions: torch.Tensor
    numeric_features: torch.Tensor
    selected_labels: torch.Tensor
    disputed_labels: torch.Tensor
    hyperedge_relation_mask: torch.Tensor
    entity_count: int
    relation_count: int

    def validate(self) -> None:
        query_count = len(self.query_keys)
        candidate_count = int(self.candidate_transitions.shape[0])
        if self.query_relations.shape != (query_count, 3):
            raise ValueError("query_relations 必须为 [query_count, 3]")
        if self.query_topics.shape != (query_count,):
            raise ValueError("query_topics 必须与查询数一致")
        if self.query_splits.shape != (query_count,):
            raise ValueError("query_splits 必须与查询数一致")
        if len(self.query_answer_ids) != query_count:
            raise ValueError("每个查询必须有一组答案")
        if self.query_offsets.shape != (query_count + 1,):
            raise ValueError("query_offsets 长度必须为查询数加一")
        if int(self.query_offsets[-1]) != candidate_count:
            raise ValueError("最后一个 query offset 必须等于候选数")
        if self.candidate_query_indices.shape != (candidate_count,):
            raise ValueError("candidate_query_indices 必须与候选数一致")
        if self.candidate_transitions.shape[1:] != (3,):
            raise ValueError("candidate_transitions 必须为 [candidate_count, 3]")
        if self.numeric_features.shape[0] != candidate_count:
            raise ValueError("numeric_features 必须与候选数一致")
        for labels in (self.selected_labels, self.disputed_labels):
            if labels.shape != (candidate_count,) or labels.dtype != torch.bool:
                raise ValueError("候选标签必须是一维布尔张量")
        if torch.any(self.selected_labels & self.disputed_labels):
            raise ValueError("选中正例不能同时是争议负例")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return dict(self.__dict__)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StructuredExperimentData":
        data = cls(**dict(payload))
        data.validate()
        return data

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), path)

    @classmethod
    def load(cls, path: Path) -> "StructuredExperimentData":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_payload(payload)


def _query_splits(query_count: int, seed: int) -> torch.Tensor:
    order = list(range(query_count))
    random.Random(seed).shuffle(order)
    train_end = int(query_count * 0.70)
    validation_end = train_end + int(query_count * 0.15)
    splits = torch.full((query_count,), TEST_SPLIT, dtype=torch.uint8)
    splits[order[:train_end]] = TRAIN_SPLIT
    splits[order[train_end:validation_end]] = VALIDATION_SPLIT
    return splits


def _relation_mask(domain: WikiTopicsDomain, relation_count: int, entity_count: int) -> torch.Tensor:
    mask = torch.zeros((entity_count, relation_count), dtype=torch.float32)
    for owner_id, relation_ids in domain.hyperedge_relations.items():
        if relation_ids:
            mask[owner_id, list(relation_ids)] = 1.0
    return mask


def _numeric_features(
    domain: WikiTopicsDomain,
    source_distances: Mapping[tuple[str, int], int],
    transition: tuple[int, int, int],
    max_degree_log: float,
) -> tuple[float, ...]:
    head_id, edge_id, tail_id = transition
    head = entity_node(head_id)
    edge = hyperedge_node(edge_id)
    tail = entity_node(tail_id)
    head_distance = source_distances[head]
    edge_distance = source_distances[edge]
    tail_distance = source_distances[tail]
    arity = domain.graph.degree[edge]
    head_degree = domain.graph.degree[head]
    tail_degree = domain.graph.degree[tail]
    distance_scale = 3 * LOGICAL_TRANSITION_COST
    return (
        head_distance / distance_scale,
        edge_distance / distance_scale,
        tail_distance / distance_scale,
        (tail_distance - head_distance) / distance_scale,
        math.log1p(arity) / max_degree_log,
        math.log1p(head_degree) / max_degree_log,
        math.log1p(tail_degree) / max_degree_log,
        float(head_id == edge_id),
        float(tail_id == edge_id),
    )


def build_structured_experiment_data(
    domain_dir: Path,
    *,
    sampler_seed: int,
    split_seed: int = 20260919,
    progress_every: int = 1000,
    max_queries: int | None = None,
) -> StructuredExperimentData:
    if max_queries is not None and max_queries <= 0:
        raise ValueError("max_queries 必须为正整数")
    domain = load_domain(domain_dir)
    examples = domain.examples[:max_queries] if max_queries is not None else domain.examples
    entity_ids = [node[1] for node in domain.graph if node[0] != HYPEREDGE_KIND]
    relation_ids = [
        relation
        for relations in domain.hyperedge_relations.values()
        for relation in relations
    ]
    relation_ids.extend(
        relation
        for example in examples
        for relation in example.query.relation_ids
    )
    entity_count = max(entity_ids) + 1
    relation_count = max(relation_ids) + 1
    max_degree_log = math.log1p(max(dict(domain.graph.degree()).values()))
    sampler_rng = random.Random(sampler_seed)

    query_keys: list[str] = []
    query_relations: list[tuple[int, int, int]] = []
    query_topics: list[int] = []
    query_answer_ids: list[tuple[int, ...]] = []
    query_offsets = [0]
    candidate_query_indices: list[int] = []
    candidate_transitions: list[tuple[int, int, int]] = []
    numeric_features: list[tuple[float, ...]] = []
    selected_labels: list[bool] = []
    disputed_labels: list[bool] = []

    for query_index, example in enumerate(examples):
        query_graph = build_query_graph(
            domain.graph, example.query.topic_id, example.answer_ids
        )
        positives = query_graph.selected_positives
        negatives = simulate_original_sampler(
            query_graph.subgraph,
            positives,
            len(positives),
            rng=sampler_rng,
        )
        transitions = tuple(positives) + tuple(negatives)
        disputed = set(
            disputed_shortest_path_transitions(
                domain.graph,
                example.query.topic_id,
                query_graph.reachable_answers,
                transitions,
                source_distances=query_graph.source_distances,
            )
        )
        positives_set = set(positives)
        query_keys.append(f"{domain.name}:{query_index}")
        query_relations.append(example.query.relation_ids)
        query_topics.append(example.query.topic_id)
        query_answer_ids.append(query_graph.reachable_answers)
        for transition in transitions:
            candidate_query_indices.append(query_index)
            candidate_transitions.append(transition)
            numeric_features.append(
                _numeric_features(
                    domain,
                    query_graph.source_distances,
                    transition,
                    max_degree_log,
                )
            )
            selected_labels.append(transition in positives_set)
            disputed_labels.append(transition in disputed and transition not in positives_set)
        query_offsets.append(len(candidate_transitions))
        if progress_every and (query_index + 1) % progress_every == 0:
            print(
                f"[{domain.name}] {query_index + 1}/{len(examples)} queries, "
                f"{len(candidate_transitions)} candidates",
                flush=True,
            )

    data = StructuredExperimentData(
        domain=domain.name,
        sampler_seed=sampler_seed,
        query_keys=query_keys,
        query_relations=torch.tensor(query_relations, dtype=torch.long),
        query_topics=torch.tensor(query_topics, dtype=torch.long),
        query_splits=_query_splits(len(query_keys), split_seed),
        query_answer_ids=query_answer_ids,
        query_offsets=torch.tensor(query_offsets, dtype=torch.long),
        candidate_query_indices=torch.tensor(candidate_query_indices, dtype=torch.long),
        candidate_transitions=torch.tensor(candidate_transitions, dtype=torch.long),
        numeric_features=torch.tensor(numeric_features, dtype=torch.float32),
        selected_labels=torch.tensor(selected_labels, dtype=torch.bool),
        disputed_labels=torch.tensor(disputed_labels, dtype=torch.bool),
        hyperedge_relation_mask=_relation_mask(domain, relation_count, entity_count),
        entity_count=entity_count,
        relation_count=relation_count,
    )
    data.validate()
    return data


def strategy_targets(
    data: StructuredExperimentData,
    strategy: str,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return labels and loss masks without changing candidates or features."""

    if strategy not in STRATEGIES:
        raise ValueError(f"未知策略 {strategy!r}")
    labels = data.selected_labels.to(dtype=torch.float32).clone()
    loss_mask = torch.ones_like(data.selected_labels)
    if strategy == "strategy2_ignore":
        loss_mask[data.disputed_labels] = False
    elif strategy == "strategy3_positive":
        labels[data.disputed_labels] = 1.0
    elif strategy == "random_drop":
        for query_index, query_key in enumerate(data.query_keys):
            start = int(data.query_offsets[query_index])
            stop = int(data.query_offsets[query_index + 1])
            drop_count = int(data.disputed_labels[start:stop].sum())
            if not drop_count:
                continue
            local_negatives = torch.where(~data.selected_labels[start:stop])[0].tolist()
            rng = random.Random()
            rng.seed(f"{seed}\0{query_key}", version=2)
            for local_index in rng.sample(local_negatives, drop_count):
                loss_mask[start + local_index] = False
    return labels, loss_mask
