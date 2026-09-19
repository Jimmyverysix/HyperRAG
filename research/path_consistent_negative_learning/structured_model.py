"""Small MLP retriever for the structured Gate C proxy experiment."""

from __future__ import annotations

import torch
from torch import nn


class StructuredRetriever(nn.Module):
    """Encode query relations, entities, fact relations, and fixed graph features."""

    def __init__(
        self,
        *,
        entity_count: int,
        relation_count: int,
        hyperedge_relation_mask: torch.Tensor,
        numeric_feature_count: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.entity_embedding = nn.Embedding(entity_count, embedding_dim)
        self.relation_embedding = nn.Embedding(relation_count, embedding_dim)
        self.register_buffer(
            "hyperedge_relation_mask",
            hyperedge_relation_mask.to(dtype=torch.float32),
        )
        input_dim = 8 * embedding_dim + numeric_feature_count
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        entity_ids: torch.Tensor,
        query_relation_ids: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        entity_features = self.entity_embedding(entity_ids).flatten(start_dim=1)
        query_features = self.relation_embedding(query_relation_ids).flatten(start_dim=1)
        owner_ids = entity_ids[:, 2]
        relation_mask = self.hyperedge_relation_mask[owner_ids]
        relation_count = relation_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        fact_features = relation_mask @ self.relation_embedding.weight / relation_count
        features = torch.cat(
            [entity_features, query_features, fact_features, numeric_features],
            dim=1,
        )
        return self.predictor(features).squeeze(1)

