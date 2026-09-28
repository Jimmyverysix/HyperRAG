"""Score fixed retrieve-only candidates and compute answer-path metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .embeddings import EmbeddingStore
from .metrics import answer_path_metrics, mean_metrics, rank_transitions
from .official import create_official_mlp
from .prepared import PreparedCandidates, assemble_features


def score_candidates(
    data: PreparedCandidates,
    embeddings: EmbeddingStore,
    checkpoint_path: Path,
    *,
    device_name: str,
    batch_size: int = 1024,
) -> torch.Tensor:
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_official_mlp(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scores = torch.empty(len(data.dde_features), dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(scores), batch_size):
            stop = min(start + batch_size, len(scores))
            indices = torch.arange(start, stop, dtype=torch.long)
            features = assemble_features(
                data,
                embeddings.node_embeddings,
                embeddings.query_embeddings,
                indices,
                device=device,
            )
            scores[start:stop] = model(features).squeeze(1).cpu()
    return scores


def evaluate_scores(
    data: PreparedCandidates,
    scores: torch.Tensor,
) -> dict[str, Any]:
    if data.transitions is None:
        raise ValueError("evaluation data has no candidate transitions")
    if scores.shape != (len(data.transitions),):
        raise ValueError("scores must align with evaluation candidates")
    query_results = []
    metric_values = []
    for query_index, query_key in enumerate(data.query_keys):
        start = int(data.query_offsets[query_index])
        stop = int(data.query_offsets[query_index + 1])
        ranked = rank_transitions(
            data.transitions[start:stop],
            scores[start:stop].tolist(),
        )
        metrics = answer_path_metrics(
            ranked,
            data.query_topics[query_index],
            data.query_answers[query_index],
        )
        metric_values.append(metrics)
        query_results.append(
            {
                "query_key": query_key,
                "candidate_count": stop - start,
                "first_answer_rank": metrics.first_answer_rank,
                "reciprocal_rank": metrics.reciprocal_rank,
                "answer_reach_5": metrics.reach_at_5,
                "answer_reach_10": metrics.reach_at_10,
            }
        )
    return {
        "domain": data.domain,
        "split": data.split,
        "query_count": len(data.query_keys),
        "candidate_count": len(data.transitions),
        "metrics": mean_metrics(metric_values),
        "queries": query_results,
    }


def candidate_path_coverage(data: PreparedCandidates) -> dict[str, float | int]:
    """Report the answer-path ceiling imposed by fixed candidates alone."""

    if data.transitions is None:
        raise ValueError("evaluation data has no candidate transitions")
    reachable = 0
    for query_index in range(len(data.query_keys)):
        start = int(data.query_offsets[query_index])
        stop = int(data.query_offsets[query_index + 1])
        metrics = answer_path_metrics(
            data.transitions[start:stop],
            data.query_topics[query_index],
            data.query_answers[query_index],
        )
        reachable += metrics.first_answer_rank is not None
    return {
        "path_reachable_query_count": reachable,
        "path_reachable_query_rate": reachable / len(data.query_keys),
        "mean_candidate_count": len(data.transitions) / len(data.query_keys),
    }
