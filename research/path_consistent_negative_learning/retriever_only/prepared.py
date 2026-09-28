"""Tensor stores shared by Retriever-only training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Mapping

import torch


@dataclass
class PreparedCandidates:
    domain: str
    split: str
    seed: int | None
    query_keys: list[str]
    query_topics: list[str]
    query_answers: list[tuple[str, ...]]
    query_offsets: torch.Tensor
    query_embedding_indices: torch.Tensor
    head_embedding_indices: torch.Tensor
    edge_embedding_indices: torch.Tensor
    tail_embedding_indices: torch.Tensor
    dde_features: torch.Tensor
    labels: torch.Tensor | None = None
    path_consistent_mask: torch.Tensor | None = None
    transitions: list[tuple[str, str, str]] | None = None
    query_multiple_shortest: list[bool] | None = None

    def validate(self) -> None:
        query_count = len(self.query_keys)
        candidate_count = int(self.dde_features.shape[0])
        if self.query_offsets.shape != (query_count + 1,):
            raise ValueError("query_offsets must have query_count + 1 values")
        if int(self.query_offsets[-1]) != candidate_count:
            raise ValueError("final query offset must equal candidate count")
        for values in (
            self.query_embedding_indices,
            self.head_embedding_indices,
            self.edge_embedding_indices,
            self.tail_embedding_indices,
        ):
            if values.shape != (candidate_count,) or values.dtype != torch.long:
                raise ValueError("embedding indices must be one-dimensional long tensors")
        if self.dde_features.shape != (candidate_count, 30):
            raise ValueError("DDE features must have shape [candidate_count, 30]")
        if len(self.query_topics) != query_count or len(self.query_answers) != query_count:
            raise ValueError("query metadata must align with query keys")
        if self.labels is not None and self.labels.shape != (candidate_count,):
            raise ValueError("labels must align with candidates")
        if self.path_consistent_mask is not None:
            if self.path_consistent_mask.shape != (candidate_count,):
                raise ValueError("path-consistent mask must align with candidates")
            if self.labels is None or torch.any(self.path_consistent_mask & self.labels.bool()):
                raise ValueError("path-consistent candidates must be negative labels")
        if self.transitions is not None and len(self.transitions) != candidate_count:
            raise ValueError("transitions must align with candidates")
        if (
            self.query_multiple_shortest is not None
            and len(self.query_multiple_shortest) != query_count
        ):
            raise ValueError("shortest-path flags must align with queries")

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": 1, **self.__dict__}, path)

    @classmethod
    def load(cls, path: Path) -> "PreparedCandidates":
        payload: Mapping[str, Any] = torch.load(
            path, map_location="cpu", weights_only=False
        )
        if payload.get("schema_version") != 1:
            raise ValueError(f"unsupported prepared candidate store: {path}")
        value = cls(**{key: item for key, item in payload.items() if key != "schema_version"})
        value.validate()
        return value


def method_weights(
    data: PreparedCandidates,
    *,
    method: str,
    lambda_: float,
    seed: int,
) -> torch.Tensor:
    """Create baseline, path-aware, or per-query matched-random weights."""

    if data.labels is None or data.path_consistent_mask is None:
        raise ValueError("training labels and path-consistent mask are required")
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    weights = torch.ones_like(data.labels, dtype=torch.float32)
    if method == "baseline":
        return weights
    if method == "ours":
        weights[data.path_consistent_mask] = lambda_
        return weights
    if method != "matched_random":
        raise ValueError(f"unsupported method: {method}")
    for query_index, query_key in enumerate(data.query_keys):
        start = int(data.query_offsets[query_index])
        stop = int(data.query_offsets[query_index + 1])
        count = int(data.path_consistent_mask[start:stop].sum())
        if count == 0:
            continue
        negative = torch.where(~data.labels[start:stop].bool())[0].tolist()
        rng = random.Random(f"{seed}|{query_key}")
        for local_index in rng.sample(negative, count):
            weights[start + local_index] = lambda_
    return weights


def assemble_features(
    data: PreparedCandidates,
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Materialize only one batch of the official 4xGTE + DDE input."""

    indices = indices.cpu()
    return torch.cat(
        (
            query_embeddings[data.query_embedding_indices[indices]],
            node_embeddings[data.head_embedding_indices[indices]],
            node_embeddings[data.edge_embedding_indices[indices]],
            node_embeddings[data.tail_embedding_indices[indices]],
            data.dde_features[indices],
        ),
        dim=1,
    ).to(device, non_blocking=True)
