"""Official GTE text embeddings and compact indexed storage."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.nn import functional as functional


@dataclass
class EmbeddingStore:
    node_names: tuple[str, ...]
    node_embeddings: torch.Tensor
    query_texts: tuple[str, ...]
    query_embeddings: torch.Tensor

    def validate(self) -> None:
        if self.node_embeddings.shape != (len(self.node_names), 1024):
            raise ValueError("node_embeddings must have shape [node_count, 1024]")
        if self.query_embeddings.shape != (len(self.query_texts), 1024):
            raise ValueError("query_embeddings must have shape [query_count, 1024]")
        if len(set(self.node_names)) != len(self.node_names):
            raise ValueError("node_names must be unique")
        if len(set(self.query_texts)) != len(self.query_texts):
            raise ValueError("query_texts must be unique")

    @cached_property
    def node_index(self) -> Mapping[str, int]:
        return {value: index for index, value in enumerate(self.node_names)}

    @cached_property
    def query_index(self) -> Mapping[str, int]:
        return {value: index for index, value in enumerate(self.query_texts)}

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "node_names": self.node_names,
                "node_embeddings": self.node_embeddings.cpu(),
                "query_texts": self.query_texts,
                "query_embeddings": self.query_embeddings.cpu(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "EmbeddingStore":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1:
            raise ValueError(f"unsupported embedding store: {path}")
        value = cls(
            node_names=tuple(payload["node_names"]),
            node_embeddings=payload["node_embeddings"].float(),
            query_texts=tuple(payload["query_texts"]),
            query_embeddings=payload["query_embeddings"].float(),
        )
        value.validate()
        return value


class OfficialGTEEncoder:
    """Match HyperRetriever's CLS pooling and L2 normalization exactly."""

    def __init__(self, model_path: str | Path, device: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        ).to(self.device)
        self.model.eval()

    def encode(self, texts: Sequence[str], *, batch_size: int) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 1024), dtype=torch.float32)
        output = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                hidden = self.model(**encoded)[0][:, 0]
                output.append(functional.normalize(hidden, p=2, dim=1).cpu())
        embeddings = torch.cat(output, dim=0).float()
        if embeddings.shape[1] != 1024:
            raise ValueError(f"GTE embedding dimension is {embeddings.shape[1]}, expected 1024")
        return embeddings


def encode_store(
    node_texts: Mapping[str, str],
    query_texts: Sequence[str],
    encoder: OfficialGTEEncoder,
    *,
    batch_size: int,
) -> EmbeddingStore:
    node_names = tuple(sorted(node_texts))
    unique_queries = tuple(sorted(set(query_texts)))
    return EmbeddingStore(
        node_names=node_names,
        node_embeddings=encoder.encode(
            [node_texts[name] for name in node_names], batch_size=batch_size
        ),
        query_texts=unique_queries,
        query_embeddings=encoder.encode(unique_queries, batch_size=batch_size),
    )
