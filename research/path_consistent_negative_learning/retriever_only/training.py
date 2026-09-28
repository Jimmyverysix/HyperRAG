"""Train the released two-layer MLP with path-aware sample weights."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn import functional as functional
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ..path_supervision.weighted_loss import weighted_binary_cross_entropy
from .embeddings import EmbeddingStore
from .official import create_official_mlp
from .prepared import PreparedCandidates, assemble_features, method_weights


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.0001
    maximum_epochs: int = 50
    patience: int = 10
    minimum_delta: float = 0.00001


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_candidate_split(
    labels: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = np.arange(len(labels))
    train, validation = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels.numpy().astype(np.int8),
        random_state=seed,
    )
    return torch.from_numpy(train).long(), torch.from_numpy(validation).long()


def _loader(indices: torch.Tensor, *, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def train_retriever(
    data: PreparedCandidates,
    embeddings: EmbeddingStore,
    *,
    method: str,
    lambda_: float,
    seed: int,
    device_name: str,
    checkpoint_path: Path,
    config: TrainingConfig = TrainingConfig(),
) -> dict[str, Any]:
    if data.labels is None:
        raise ValueError("training data has no labels")
    seed_everything(seed)
    device = torch.device(device_name)
    weights = method_weights(
        data,
        method=method,
        lambda_=lambda_,
        seed=seed,
    )
    train_indices, validation_indices = stratified_candidate_split(
        data.labels,
        seed=seed,
    )
    train_loader = _loader(
        train_indices,
        batch_size=config.batch_size,
        seed=seed,
        shuffle=True,
    )
    validation_loader = _loader(
        validation_indices,
        batch_size=config.batch_size,
        seed=seed,
        shuffle=False,
    )
    model = create_official_mlp(device=device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    node_embeddings = embeddings.node_embeddings
    query_embeddings = embeddings.query_embeddings
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        train_weighted_loss = 0.0
        train_weight_sum = 0.0
        for (indices,) in train_loader:
            features = assemble_features(
                data,
                node_embeddings,
                query_embeddings,
                indices,
                device=device,
            )
            labels = data.labels[indices].float().to(device).unsqueeze(1)
            batch_weights = weights[indices].to(device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = weighted_binary_cross_entropy(logits, labels, batch_weights)
            loss.backward()
            optimizer.step()
            weight_sum = float(batch_weights.sum())
            train_weighted_loss += float(loss.detach()) * weight_sum
            train_weight_sum += weight_sum

        model.eval()
        validation_weighted_loss = 0.0
        validation_weight_sum = 0.0
        with torch.no_grad():
            for (indices,) in validation_loader:
                features = assemble_features(
                    data,
                    node_embeddings,
                    query_embeddings,
                    indices,
                    device=device,
                )
                labels = data.labels[indices].float().to(device).unsqueeze(1)
                batch_weights = weights[indices].to(device).unsqueeze(1)
                logits = model(features)
                elementwise = functional.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    reduction="none",
                )
                validation_weighted_loss += float((elementwise * batch_weights).sum())
                validation_weight_sum += float(batch_weights.sum())
        train_loss = train_weighted_loss / train_weight_sum
        validation_loss = validation_weighted_loss / validation_weight_sum
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss - config.minimum_delta:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": best_state,
            "pred_in_size": 4126,
            "emb_size": 256,
            "method": method,
            "lambda": lambda_,
            "seed": seed,
            "domain": data.domain,
            "training_config": config.__dict__,
        },
        checkpoint_path,
    )
    return {
        "domain": data.domain,
        "method": method,
        "lambda": lambda_,
        "seed": seed,
        "candidate_count": len(data.labels),
        "positive_count": int(data.labels.sum()),
        "path_consistent_negative_count": int(data.path_consistent_mask.sum()),
        "train_candidate_count": len(train_indices),
        "internal_validation_candidate_count": len(validation_indices),
        "best_internal_validation_loss": best_loss,
        "epochs_run": len(history),
        "history": history,
    }
