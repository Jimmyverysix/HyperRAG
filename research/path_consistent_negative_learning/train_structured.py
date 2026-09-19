"""Train and evaluate one arm of the WikiTopics structured Gate C experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import RetrievalMetrics, evaluate_retrieval
from .strategies import STRATEGIES, WEIGHTED_STRATEGIES
from .structured_data import (
    SELECTION_SPLIT,
    SPLIT_NAMES,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    StructuredExperimentData,
    strategy_loss_weights,
)
from .structured_model import StructuredRetriever


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _candidate_indices_for_split(
    data: StructuredExperimentData,
    split: int,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    query_mask = data.query_splits[data.candidate_query_indices] == split
    if loss_mask is not None:
        query_mask &= loss_mask
    return torch.where(query_mask)[0]


def _batch_inputs(
    data: StructuredExperimentData,
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_indices = data.candidate_query_indices[indices]
    topics = data.query_topics[query_indices].unsqueeze(1)
    transitions = data.candidate_transitions[indices]
    entity_ids = torch.cat([topics, transitions], dim=1).to(device, non_blocking=True)
    query_relations = data.query_relations[query_indices].to(device, non_blocking=True)
    numeric = data.numeric_features[indices].to(device, non_blocking=True)
    return entity_ids, query_relations, numeric


def _mean_loss(
    model: StructuredRetriever,
    data: StructuredExperimentData,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0
    total_weight = 0.0
    loader = DataLoader(TensorDataset(indices), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_indices,) in loader:
            entity_ids, query_relations, numeric = _batch_inputs(
                data, batch_indices, device
            )
            logits = model(entity_ids, query_relations, numeric)
            batch_labels = labels[batch_indices].to(device, non_blocking=True)
            batch_weights = loss_weights[batch_indices].to(device, non_blocking=True)
            total_loss += float((loss_function(logits, batch_labels) * batch_weights).sum())
            total_weight += float(batch_weights.sum())
    if total_weight <= 0.0:
        raise ValueError("损失权重之和必须大于零")
    return total_loss / total_weight


def weighted_binary_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    normalization_weight: float | None = None,
) -> torch.Tensor:
    """Compute BCE normalized by effective supervision weight."""

    denominator = weights.sum()
    if float(denominator.detach()) <= 0.0:
        raise ValueError("损失权重之和必须大于零")
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
    )
    if normalization_weight is None:
        return (losses * weights).sum() / denominator
    if normalization_weight <= 0.0:
        raise ValueError("全局平均损失权重必须大于零")
    return (losses * weights).mean() / normalization_weight


def _score_indices(
    model: StructuredRetriever,
    data: StructuredExperimentData,
    indices: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    scores = torch.empty(len(data.selected_labels), dtype=torch.float32)
    loader = DataLoader(TensorDataset(indices), batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        for (batch_indices,) in loader:
            entity_ids, query_relations, numeric = _batch_inputs(
                data, batch_indices, device
            )
            scores[batch_indices] = model(entity_ids, query_relations, numeric).cpu()
    return scores


def _metrics_dict(metrics: RetrievalMetrics) -> dict[str, Any]:
    return {
        "query_count": metrics.query_count,
        "evaluated_query_count": metrics.evaluated_query_count,
        "skipped_query_count": metrics.skipped_query_count,
        "candidate_count": metrics.candidate_count,
        "positive_count": metrics.positive_count,
        "mrr": metrics.mrr,
        "hits_at": metrics.hits_at,
        "recall_at": metrics.recall_at,
        "micro_recall_at": metrics.micro_recall_at,
        "question_pr_auc": metrics.question_pr_auc,
        "candidate_pr_auc": metrics.candidate_pr_auc,
    }


def _query_metric_rows(
    data: StructuredExperimentData,
    scores: torch.Tensor,
    selected_metrics: RetrievalMetrics,
    all_shortest_metrics: RetrievalMetrics,
    answer_reach: dict[int, dict[str, float]],
    split: int = TEST_SPLIT,
) -> list[dict[str, Any]]:
    rows = []
    for query_index, query_key in enumerate(data.query_keys):
        if int(data.query_splits[query_index]) != split:
            continue
        start = int(data.query_offsets[query_index])
        stop = int(data.query_offsets[query_index + 1])
        if start == stop:
            continue
        selected = selected_metrics.per_query[query_key]
        all_shortest = all_shortest_metrics.per_query[query_key]
        if selected is None or all_shortest is None:
            continue
        row: dict[str, Any] = {
            "query_key": query_key,
            "selected_mrr": selected.reciprocal_rank,
            "selected_pr_auc": selected.pr_auc,
            "all_shortest_mrr": all_shortest.reciprocal_rank,
            "all_shortest_pr_auc": all_shortest.pr_auc,
        }
        for cutoff, value in selected.hits_at.items():
            row[f"selected_hits_{cutoff}"] = value
        for cutoff, value in selected.recall_at.items():
            row[f"selected_recall_{cutoff}"] = value
        for cutoff, value in all_shortest.hits_at.items():
            row[f"all_shortest_hits_{cutoff}"] = value
        for cutoff, value in all_shortest.recall_at.items():
            row[f"all_shortest_recall_{cutoff}"] = value
        for cutoff, values in answer_reach.items():
            row[f"answer_reach_{cutoff}"] = values[query_key]
        rows.append(row)
    return rows


def _answer_reachability(
    data: StructuredExperimentData,
    scores: torch.Tensor,
    cutoffs: Iterable[int],
    split: int = TEST_SPLIT,
) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
    cutoffs = tuple(sorted(set(cutoffs)))
    per_query = {cutoff: {} for cutoff in cutoffs}
    for query_index, query_key in enumerate(data.query_keys):
        if int(data.query_splits[query_index]) != split:
            continue
        start = int(data.query_offsets[query_index])
        stop = int(data.query_offsets[query_index + 1])
        if start == stop or not data.query_answer_ids[query_index]:
            continue
        local_scores = scores[start:stop].numpy()
        order = np.argsort(-local_scores, kind="stable")
        transitions = data.candidate_transitions[start:stop].numpy()
        topic = int(data.query_topics[query_index])
        answers = set(data.query_answer_ids[query_index])
        for cutoff in cutoffs:
            graph = nx.DiGraph()
            graph.add_edges_from(
                (int(transitions[index, 0]), int(transitions[index, 2]))
                for index in order[:cutoff]
            )
            reachable = nx.descendants(graph, topic) if topic in graph else set()
            per_query[cutoff][query_key] = float(bool(reachable & answers))
    aggregate = {
        cutoff: float(np.mean(list(values.values()))) if values else 0.0
        for cutoff, values in per_query.items()
    }
    return aggregate, per_query


def evaluate_split(
    model: StructuredRetriever,
    data: StructuredExperimentData,
    *,
    device: torch.device,
    batch_size: int,
    split: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluation_indices = _candidate_indices_for_split(data, split)
    scores = _score_indices(
        model, data, evaluation_indices, device=device, batch_size=batch_size
    )
    selected_queries = {}
    all_shortest_queries = {}
    for query_index, query_key in enumerate(data.query_keys):
        if int(data.query_splits[query_index]) != split:
            continue
        start = int(data.query_offsets[query_index])
        stop = int(data.query_offsets[query_index + 1])
        if start == stop:
            continue
        query_scores = scores[start:stop].tolist()
        selected = data.selected_labels[start:stop].tolist()
        all_shortest = (
            data.selected_labels[start:stop] | data.disputed_labels[start:stop]
        ).tolist()
        selected_queries[query_key] = (query_scores, selected)
        all_shortest_queries[query_key] = (query_scores, all_shortest)
    selected_metrics = evaluate_retrieval(
        selected_queries, ks=(1, 3, 5, 10, 20), no_positive="skip"
    )
    all_shortest_metrics = evaluate_retrieval(
        all_shortest_queries, ks=(1, 3, 5, 10, 20), no_positive="skip"
    )
    answer_reach, answer_reach_per_query = _answer_reachability(
        data, scores, cutoffs=(3, 5, 10, 20), split=split
    )
    aggregate = {
        "selected_path": _metrics_dict(selected_metrics),
        "all_shortest_paths": _metrics_dict(all_shortest_metrics),
        "answer_reach_at": answer_reach,
    }
    rows = _query_metric_rows(
        data,
        scores,
        selected_metrics,
        all_shortest_metrics,
        answer_reach_per_query,
        split=split,
    )
    return aggregate, rows


def evaluate_test(
    model: StructuredRetriever,
    data: StructuredExperimentData,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compatibility wrapper for the original Gate C evaluator."""

    return evaluate_split(
        model,
        data,
        device=device,
        batch_size=batch_size,
        split=TEST_SPLIT,
    )


def train(
    *,
    data_path: Path,
    strategy: str,
    seed: int,
    output_dir: Path,
    device_name: str = "cuda",
    epochs: int = 50,
    patience: int = 8,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    disputed_negative_weight: float | None = None,
    evaluation_split: int = TEST_SPLIT,
    experiment_name: str = "structured_proxy_gate_c",
) -> dict[str, Any]:
    if strategy not in STRATEGIES + WEIGHTED_STRATEGIES:
        raise ValueError(f"未知策略 {strategy!r}")
    if evaluation_split not in SPLIT_NAMES:
        raise ValueError(f"未知评估划分：{evaluation_split}")
    if strategy in WEIGHTED_STRATEGIES and disputed_negative_weight is None:
        raise ValueError(f"{strategy} 需要 disputed_negative_weight")
    if strategy not in WEIGHTED_STRATEGIES and disputed_negative_weight is not None:
        raise ValueError(f"{strategy} 不接受 disputed_negative_weight")
    set_seed(seed)
    data = StructuredExperimentData.load(data_path)
    labels, loss_weights = strategy_loss_weights(
        data,
        strategy,
        seed=seed,
        disputed_negative_weight=disputed_negative_weight,
    )
    device = torch.device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = StructuredRetriever(
        entity_count=data.entity_count,
        relation_count=data.relation_count,
        hyperedge_relation_mask=data.hyperedge_relation_mask,
        numeric_feature_count=data.numeric_features.shape[1],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positive_weight_mask = loss_weights > 0.0
    train_indices = _candidate_indices_for_split(
        data,
        TRAIN_SPLIT,
        positive_weight_mask,
    )
    validation_indices = _candidate_indices_for_split(
        data,
        VALIDATION_SPLIT,
        positive_weight_mask,
    )
    mean_training_weight = float(loss_weights[train_indices].mean())
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float | int]] = []
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        loader = DataLoader(
            TensorDataset(train_indices),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for (batch_indices,) in loader:
            entity_ids, query_relations, numeric = _batch_inputs(
                data, batch_indices, device
            )
            batch_labels = labels[batch_indices].to(device, non_blocking=True)
            batch_weights = loss_weights[batch_indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(entity_ids, query_relations, numeric)
            loss = weighted_binary_cross_entropy(
                logits,
                batch_labels,
                batch_weights,
                normalization_weight=mean_training_weight,
            )
            loss.backward()
            optimizer.step()
            batch_weight = float(batch_weights.sum())
            train_loss_sum += (
                loss.item() * len(batch_indices) * mean_training_weight
            )
            train_items += batch_weight
        train_loss = train_loss_sum / train_items
        validation_loss = _mean_loss(
            model,
            data,
            labels,
            loss_weights,
            validation_indices,
            device=device,
            batch_size=batch_size,
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
        if validation_loss < best_validation - 1e-5:
            best_validation = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(
        torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    metrics, query_rows = evaluate_split(
        model,
        data,
        device=device,
        batch_size=batch_size,
        split=evaluation_split,
    )
    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    result = {
        "schema_version": 2,
        "experiment": experiment_name,
        "domain": data.domain,
        "strategy": strategy,
        "seed": seed,
        "sampler_seed": data.sampler_seed,
        "split_seed": data.split_seed,
        "split_scheme": data.split_scheme,
        "evaluation_split": SPLIT_NAMES[evaluation_split],
        "disputed_negative_weight": disputed_negative_weight,
        "candidate_count": len(data.selected_labels),
        "selected_positive_count": int(data.selected_labels.sum()),
        "disputed_negative_count": int(data.disputed_labels.sum()),
        "active_training_candidate_count": int(
            positive_weight_mask[_candidate_indices_for_split(data, TRAIN_SPLIT)].sum()
        ),
        "effective_training_weight": float(
            loss_weights[_candidate_indices_for_split(data, TRAIN_SPLIT)].sum()
        ),
        "mean_training_weight": mean_training_weight,
        "training_config": {
            "epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "training_seconds": elapsed,
        "peak_cuda_memory_bytes": peak_memory,
        "metrics": metrics,
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "query_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in query_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "train_loss", "validation_loss"]
        )
        writer.writeheader()
        writer.writerows(history)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES + WEIGHTED_STRATEGIES,
        required=True,
    )
    parser.add_argument("--experiment-name", default="structured_proxy_gate_c")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--disputed-negative-weight", type=float)
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "selection", "test"),
        default="test",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train(
        data_path=args.data,
        strategy=args.strategy,
        seed=args.seed,
        output_dir=args.output_dir,
        device_name=args.device,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        disputed_negative_weight=args.disputed_negative_weight,
        evaluation_split={
            "validation": VALIDATION_SPLIT,
            "selection": SELECTION_SPLIT,
            "test": TEST_SPLIT,
        }[args.evaluation_split],
        experiment_name=args.experiment_name,
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
