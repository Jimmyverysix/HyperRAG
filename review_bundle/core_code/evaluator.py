"""Reviewer snapshot of production source files; not a standalone module."""

# ===== BEGIN PRODUCTION SOURCE: research/path_consistent_negative_learning/retriever_only/metrics.py =====
"""Answer-path metrics for raw Retriever candidate logits."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Mapping, Sequence

from .graph import Transition


@dataclass(frozen=True)
class AnswerPathMetrics:
    reciprocal_rank: float
    first_answer_rank: int | None
    reach_at_5: float
    reach_at_10: float


def rank_transitions(
    transitions: Sequence[Transition],
    scores: Sequence[float],
) -> tuple[Transition, ...]:
    if len(transitions) != len(scores):
        raise ValueError("transitions and scores must have equal length")
    return tuple(
        transition
        for transition, _ in sorted(
            zip(transitions, scores),
            key=lambda item: (-float(item[1]), item[0]),
        )
    )


def answer_path_metrics(
    ranked_transitions: Sequence[Transition],
    topic: str,
    answers: Sequence[str],
) -> AnswerPathMetrics:
    """Find when incrementally inserted directed transitions reach an answer."""

    answer_set = set(answers)
    if topic in answer_set:
        first_rank: int | None = 1
    else:
        first_rank = None
    adjacency: dict[str, set[str]] = {}
    reachable = {topic}
    for rank, (head, _, tail) in enumerate(ranked_transitions, start=1):
        adjacency.setdefault(head, set()).add(tail)
        if head not in reachable or tail in reachable:
            continue
        queue = [tail]
        reachable.add(tail)
        while queue:
            current = queue.pop()
            for following in adjacency.get(current, ()):
                if following not in reachable:
                    reachable.add(following)
                    queue.append(following)
        if first_rank is None and reachable & answer_set:
            first_rank = rank
            break
    reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank
    return AnswerPathMetrics(
        reciprocal_rank=reciprocal_rank,
        first_answer_rank=first_rank,
        reach_at_5=float(first_rank is not None and first_rank <= 5),
        reach_at_10=float(first_rank is not None and first_rank <= 10),
    )


def answer_path_completion_rank_minimax(
    ranked_transitions: Sequence[Transition],
    topic: str,
    answers: Sequence[str],
) -> int | None:
    """Compute the APC rank as the minimum path bottleneck rank.

    For every directed topic--answer path, its cost is the largest rank of a
    transition on that path.  The returned value is the minimum such cost.
    This is the formal min--max definition of Answer-Path Completion rank and
    is equivalent to the incremental top-k reachability implementation above.
    """

    answer_set = set(answers)
    if topic in answer_set:
        return 1
    adjacency: dict[str, list[tuple[str, int]]] = {}
    for rank, (head, _, tail) in enumerate(ranked_transitions, start=1):
        adjacency.setdefault(head, []).append((tail, rank))
    best = {topic: 0}
    queue: list[tuple[int, str]] = [(0, topic)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != best[node]:
            continue
        if node in answer_set:
            return cost
        for following, rank in adjacency.get(node, ()):
            candidate = max(cost, rank)
            if candidate < best.get(following, candidate + 1):
                best[following] = candidate
                heapq.heappush(queue, (candidate, following))
    return None


def mean_metrics(values: Sequence[AnswerPathMetrics]) -> Mapping[str, float]:
    if not values:
        raise ValueError("at least one query metric is required")
    size = len(values)
    return {
        "answer_path_mrr": sum(value.reciprocal_rank for value in values) / size,
        "answer_reach_5": sum(value.reach_at_5 for value in values) / size,
        "answer_reach_10": sum(value.reach_at_10 for value in values) / size,
    }
# ===== END PRODUCTION SOURCE: research/path_consistent_negative_learning/retriever_only/metrics.py =====

# ===== BEGIN PRODUCTION SOURCE: research/path_consistent_negative_learning/retriever_only/evaluation.py =====
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
    *,
    multiple_shortest_only: bool = False,
) -> dict[str, Any]:
    if data.transitions is None:
        raise ValueError("evaluation data has no candidate transitions")
    if scores.shape != (len(data.transitions),):
        raise ValueError("scores must align with evaluation candidates")
    if multiple_shortest_only and data.query_multiple_shortest is None:
        raise ValueError("evaluation data has no shortest-path multiplicity flags")
    query_results = []
    metric_values = []
    evaluated_candidate_count = 0
    shortest_flags = data.query_multiple_shortest
    for query_index, query_key in enumerate(data.query_keys):
        if (
            multiple_shortest_only
            and shortest_flags is not None
            and not shortest_flags[query_index]
        ):
            continue
        start = int(data.query_offsets[query_index])
        stop = int(data.query_offsets[query_index + 1])
        evaluated_candidate_count += stop - start
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
    if not metric_values:
        raise ValueError("evaluation subset contains no queries")
    return {
        "domain": data.domain,
        "split": data.split,
        "evaluation_subset": (
            "multiple_equal_shortest_paths"
            if multiple_shortest_only
            else "all_eligible_queries"
        ),
        "query_count": len(query_results),
        "candidate_count": evaluated_candidate_count,
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
# ===== END PRODUCTION SOURCE: research/path_consistent_negative_learning/retriever_only/evaluation.py =====
