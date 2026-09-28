"""Prepare indexed GTE+DDE candidates without materializing 4126-D files."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .candidates import (
    build_training_queries,
    semantic_beam_candidates,
)
from .data import AlignedQuery
from .embeddings import EmbeddingStore
from .graph import DeterministicHypergraph, Transition
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
        scores = {name: similarities[index] for name, index in node_index.items()}
        candidates = tuple(
            value.transition
            for value in semantic_beam_candidates(
                bundle.graph,
                query.topic_node,
                scores,
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
        )
        if progress_every and eligible % progress_every == 0:
            print(f"[{domain}/{split}] {eligible} queries", flush=True)
    return accumulator.finish()
