"""Reviewer snapshot of production source files; not a standalone module."""

# ===== BEGIN PRODUCTION SOURCE: research/path_consistent_negative_learning/path_supervision/weighted_loss.py =====
"""路径一致弱负例的加权二元交叉熵。"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def weighted_binary_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """计算 ``sum(w_i BCE_i) / sum(w_i)``。

    归一化只使用当前输入中的有效监督权重，因此 ``lambda=0`` 与删除对应
    样本完全等价，``lambda=1`` 与普通 mean BCE 完全等价。
    """

    if logits.shape != labels.shape or logits.shape != weights.shape:
        raise ValueError("logits、labels 与 weights 必须具有相同形状")
    if not logits.is_floating_point() or not labels.is_floating_point():
        raise TypeError("logits 与 labels 必须是浮点张量")
    if not weights.is_floating_point():
        raise TypeError("weights 必须是浮点张量")
    if torch.any(weights < 0):
        raise ValueError("损失权重不能为负")

    denominator = weights.sum()
    if denominator.detach().item() <= 0.0:
        raise ValueError("损失权重之和必须大于零")
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
    )
    return (elementwise * weights).sum() / denominator


def path_consistent_weak_negative_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    path_consistent_negative_mask: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """仅降低路径一致负例的监督权重，不改变二元标签。"""

    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda 必须位于 [0, 1]")
    if path_consistent_negative_mask.shape != labels.shape:
        raise ValueError("路径一致掩码必须与 labels 具有相同形状")
    if path_consistent_negative_mask.dtype is not torch.bool:
        raise TypeError("路径一致掩码必须是布尔张量")
    if torch.any(path_consistent_negative_mask & (labels != 0)):
        raise ValueError("只能对负例降低监督权重")

    weights = torch.ones_like(logits)
    weights.masked_fill_(path_consistent_negative_mask, lambda_)
    return weighted_binary_cross_entropy(logits, labels, weights)
# ===== END PRODUCTION SOURCE: research/path_consistent_negative_learning/path_supervision/weighted_loss.py =====

# ===== BEGIN PRODUCTION SOURCE: research/path_consistent_negative_learning/retriever_only/training.py =====
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

from ..path_supervision.weighted_loss import weighted_binary_cross_entropy
from .embeddings import EmbeddingStore
from .official import create_official_mlp
from .prepared import PreparedCandidates, materialize_features, method_weights


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.0001
    maximum_epochs: int = 50
    patience: int = 10
    minimum_delta: float = 0.00001
    feature_materialization_chunk_size: int = 8192


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


def _index_batches(
    indices: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
    shuffle: bool,
):
    """Yield the exact index batches produced by the frozen DataLoader.

    PyTorch's single-process DataLoader consumes one base-seed draw whenever
    an iterator is created.  RandomSampler then generates one full
    permutation and, because ``num_samples % n == 0``, a second discarded
    permutation.  Replaying those draws keeps every epoch's order identical
    while avoiding ``tolist()``, per-sample TensorDataset access, and collate.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    if shuffle:
        positions = torch.randperm(len(indices), generator=generator)
    else:
        positions = None
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        yield (
            indices[positions[start:stop]]
            if positions is not None
            else indices[start:stop]
        )
    if shuffle:
        # RandomSampler evaluates a second randperm(...).tolist()[:0].
        torch.randperm(len(indices), generator=generator)


def _ordered_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    """Reproduce the original ordered Python-float weighted accumulation."""

    value_list = values.detach().cpu().tolist()
    weight_list = weights.detach().cpu().tolist()
    weighted_sum = 0.0
    weight_sum = 0.0
    for value, weight in zip(value_list, weight_list):
        weighted_sum += value * weight
        weight_sum += weight
    return weighted_sum / weight_sum


def _ordered_ratio(
    numerators: torch.Tensor,
    denominators: torch.Tensor,
) -> float:
    """Reproduce the original ordered Python-float ratio accumulation."""

    numerator_sum = 0.0
    denominator_sum = 0.0
    for numerator, denominator in zip(
        numerators.detach().cpu().tolist(),
        denominators.detach().cpu().tolist(),
    ):
        numerator_sum += numerator
        denominator_sum += denominator
    return numerator_sum / denominator_sum


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
    train_generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed)
    model = create_official_mlp(device=device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    features = materialize_features(
        data,
        embeddings.node_embeddings,
        embeddings.query_embeddings,
        device=device,
        chunk_size=config.feature_materialization_chunk_size,
    )
    labels = data.labels.float().to(device).unsqueeze(1)
    device_weights = weights.to(device).unsqueeze(1)
    positive_weight_mask = weights > 0
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        train_batch_count = (
            len(train_indices) + config.batch_size - 1
        ) // config.batch_size
        train_losses = torch.empty(train_batch_count, device=device)
        train_weight_sums = torch.empty(train_batch_count, device=device)
        for batch_index, indices in enumerate(_index_batches(
            train_indices,
            batch_size=config.batch_size,
            generator=train_generator,
            shuffle=True,
        )):
            if not bool(positive_weight_mask[indices].any()):
                # With lambda=0, a shuffled batch can contain only excluded
                # samples.  Skipping the optimizer step is the exact analogue
                # of that batch carrying no supervision; an Adam step with
                # zero gradients could still move parameters via momentum.
                train_losses[batch_index] = 0.0
                train_weight_sums[batch_index] = 0.0
                continue
            device_indices = indices.to(device)
            batch_features = features[device_indices]
            batch_labels = labels[device_indices]
            batch_weights = device_weights[device_indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = weighted_binary_cross_entropy(
                logits,
                batch_labels,
                batch_weights,
            )
            loss.backward()
            optimizer.step()
            train_losses[batch_index] = loss.detach()
            train_weight_sums[batch_index] = batch_weights.sum()

        model.eval()
        validation_batch_count = (
            len(validation_indices) + config.batch_size - 1
        ) // config.batch_size
        validation_numerators = torch.empty(validation_batch_count, device=device)
        validation_weight_sums = torch.empty(validation_batch_count, device=device)
        with torch.no_grad():
            for batch_index, indices in enumerate(_index_batches(
                validation_indices,
                batch_size=config.batch_size,
                generator=validation_generator,
                shuffle=False,
            )):
                device_indices = indices.to(device)
                batch_features = features[device_indices]
                batch_labels = labels[device_indices]
                batch_weights = device_weights[device_indices]
                logits = model(batch_features)
                elementwise = functional.binary_cross_entropy_with_logits(
                    logits,
                    batch_labels,
                    reduction="none",
                )
                validation_numerators[batch_index] = (
                    elementwise * batch_weights
                ).sum()
                validation_weight_sums[batch_index] = batch_weights.sum()
        train_loss = _ordered_weighted_mean(train_losses, train_weight_sums)
        validation_loss = _ordered_ratio(
            validation_numerators,
            validation_weight_sums,
        )
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
# ===== END PRODUCTION SOURCE: research/path_consistent_negative_learning/retriever_only/training.py =====
