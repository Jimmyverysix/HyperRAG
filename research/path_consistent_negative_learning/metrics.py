"""Retrieval metrics for scored candidates grouped by query.

Ranking metrics (MRR, Hits@K, and Recall@K) are defined per query and then
macro-averaged.  ``candidate_pr_auc`` pools candidates from all evaluated
queries, while ``question_pr_auc`` macro-averages the per-query PR-AUC.  Here
PR-AUC is average precision, i.e. the step-wise area under the precision-recall
curve.  This definition is invariant to the order of candidates with tied
scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray


BinaryLabel: TypeAlias = bool | int
CandidateList: TypeAlias = tuple[Sequence[float], Sequence[BinaryLabel]]
NoPositivePolicy: TypeAlias = Literal["skip", "zero", "error"]


@dataclass(frozen=True)
class QueryRetrievalMetrics:
    """Metrics for the ranked candidates of one query."""

    candidate_count: int
    positive_count: int
    reciprocal_rank: float
    hits_at: dict[int, float]
    recall_at: dict[int, float]
    pr_auc: float


@dataclass(frozen=True)
class RetrievalMetrics:
    """Per-query and aggregate retrieval metrics.

    ``per_query`` contains ``None`` for no-positive queries when
    ``no_positive="skip"``.  MRR, Hits@K, Recall@K, and ``question_pr_auc``
    average only the evaluated queries. ``micro_recall_at`` and
    ``candidate_pr_auc`` pool their candidate-level sufficient statistics.
    """

    query_count: int
    evaluated_query_count: int
    skipped_query_count: int
    candidate_count: int
    positive_count: int
    mrr: float
    hits_at: dict[int, float]
    recall_at: dict[int, float]
    micro_recall_at: dict[int, float]
    question_pr_auc: float
    candidate_pr_auc: float
    per_query: dict[Hashable, QueryRetrievalMetrics | None]


def _validate_ks(ks: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(set(ks)))
    if not normalized:
        raise ValueError("ks must contain at least one cutoff")
    if any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in normalized):
        raise ValueError("each cutoff in ks must be a positive integer")
    return normalized


def _validate_policy(policy: str) -> NoPositivePolicy:
    if policy not in {"skip", "zero", "error"}:
        raise ValueError("no_positive must be one of: 'skip', 'zero', 'error'")
    return policy  # type: ignore[return-value]


def _as_arrays(
    scores: Sequence[float],
    labels: Sequence[BinaryLabel],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels)
    if score_array.ndim != 1 or label_array.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional")
    if score_array.shape[0] != label_array.shape[0]:
        raise ValueError("scores and labels must have the same length")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must contain only finite values")

    try:
        numeric_labels = label_array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be binary values (0/1 or bool)") from exc
    if not np.all(np.isfinite(numeric_labels)) or not np.all(
        (numeric_labels == 0.0) | (numeric_labels == 1.0)
    ):
        raise ValueError("labels must be binary values (0/1 or bool)")
    return score_array, numeric_labels.astype(np.int64)


def precision_recall_auc(
    scores: Sequence[float],
    labels: Sequence[BinaryLabel],
    *,
    no_positive: NoPositivePolicy = "zero",
) -> float | None:
    """Return average precision for binary labels.

    Equal-score candidates are evaluated at the same threshold.  If there is
    no positive label, ``no_positive`` determines whether to return ``None``,
    return ``0.0``, or raise ``ValueError``.
    """

    policy = _validate_policy(no_positive)
    score_array, label_array = _as_arrays(scores, labels)
    positive_count = int(label_array.sum())
    if positive_count == 0:
        if policy == "skip":
            return None
        if policy == "error":
            raise ValueError("PR-AUC is undefined because the labels contain no positive")
        return 0.0

    order = np.argsort(-score_array, kind="stable")
    sorted_scores = score_array[order]
    sorted_labels = label_array[order]
    true_positives = np.cumsum(sorted_labels)

    # Keep the final item at every distinct score threshold.  This makes the
    # result independent of the original ordering within score ties.
    threshold_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    threshold_tp = true_positives[threshold_ends].astype(np.float64)
    precision = threshold_tp / (threshold_ends + 1)
    recall = threshold_tp / positive_count
    recall_increase = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increase * precision))


def evaluate_query(
    scores: Sequence[float],
    labels: Sequence[BinaryLabel],
    *,
    ks: Sequence[int] = (1, 5, 10),
    no_positive: NoPositivePolicy = "zero",
) -> QueryRetrievalMetrics | None:
    """Evaluate one query's candidate scores and binary relevance labels.

    Candidates are ranked by descending score; original input order is used as
    the deterministic tie-breaker for rank-based metrics.  PR-AUC treats tied
    scores as one threshold.  See :func:`precision_recall_auc` for the
    no-positive behavior.
    """

    cutoffs = _validate_ks(ks)
    policy = _validate_policy(no_positive)
    score_array, label_array = _as_arrays(scores, labels)
    positive_count = int(label_array.sum())
    if positive_count == 0:
        if policy == "skip":
            return None
        if policy == "error":
            raise ValueError("query has no positive candidate")
        return QueryRetrievalMetrics(
            candidate_count=len(label_array),
            positive_count=0,
            reciprocal_rank=0.0,
            hits_at={k: 0.0 for k in cutoffs},
            recall_at={k: 0.0 for k in cutoffs},
            pr_auc=0.0,
        )

    order = np.argsort(-score_array, kind="stable")
    ranked_labels = label_array[order]
    positive_ranks = np.flatnonzero(ranked_labels == 1) + 1
    first_positive_rank = int(positive_ranks[0])
    hits = {k: float(first_positive_rank <= k) for k in cutoffs}
    recall = {
        k: float(np.count_nonzero(positive_ranks <= k) / positive_count)
        for k in cutoffs
    }
    pr_auc = precision_recall_auc(score_array, label_array, no_positive="error")
    assert pr_auc is not None
    return QueryRetrievalMetrics(
        candidate_count=len(label_array),
        positive_count=positive_count,
        reciprocal_rank=1.0 / first_positive_rank,
        hits_at=hits,
        recall_at=recall,
        pr_auc=pr_auc,
    )


def evaluate_retrieval(
    queries: Mapping[Hashable, CandidateList],
    *,
    ks: Sequence[int] = (1, 5, 10),
    no_positive: NoPositivePolicy = "skip",
) -> RetrievalMetrics:
    """Evaluate candidate rankings for multiple queries.

    Args:
        queries: Mapping from query key to ``(scores, labels)`` sequences.
        ks: Positive rank cutoffs.
        no_positive: How to handle a query without a positive candidate:
            ``"skip"`` excludes it, ``"zero"`` includes zero-valued metrics,
            and ``"error"`` raises ``ValueError``.

    Returns:
        Per-query results, question-level macro metrics, candidate-level pooled
        PR-AUC, and micro Recall@K.
    """

    cutoffs = _validate_ks(ks)
    policy = _validate_policy(no_positive)
    per_query: dict[Hashable, QueryRetrievalMetrics | None] = {}
    evaluated: list[QueryRetrievalMetrics] = []
    pooled_scores: list[NDArray[np.float64]] = []
    pooled_labels: list[NDArray[np.int64]] = []
    retrieved_positive_counts = {k: 0 for k in cutoffs}

    for query_key, candidate_data in queries.items():
        try:
            scores, labels = candidate_data
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"query {query_key!r} must map to a (scores, labels) pair"
            ) from exc
        score_array, label_array = _as_arrays(scores, labels)
        result = evaluate_query(
            score_array,
            label_array,
            ks=cutoffs,
            no_positive=policy,
        )
        per_query[query_key] = result
        if result is None:
            continue
        evaluated.append(result)
        pooled_scores.append(score_array)
        pooled_labels.append(label_array)
        ranked_labels = label_array[np.argsort(-score_array, kind="stable")]
        for k in cutoffs:
            retrieved_positive_counts[k] += int(ranked_labels[:k].sum())

    evaluated_count = len(evaluated)
    candidate_count = sum(result.candidate_count for result in evaluated)
    positive_count = sum(result.positive_count for result in evaluated)

    def macro(attribute: str) -> float:
        if not evaluated:
            return 0.0
        return float(np.mean([getattr(result, attribute) for result in evaluated]))

    hits_at = {
        k: float(np.mean([result.hits_at[k] for result in evaluated]))
        if evaluated
        else 0.0
        for k in cutoffs
    }
    recall_at = {
        k: float(np.mean([result.recall_at[k] for result in evaluated]))
        if evaluated
        else 0.0
        for k in cutoffs
    }
    micro_recall_at = {
        k: retrieved_positive_counts[k] / positive_count if positive_count else 0.0
        for k in cutoffs
    }
    if pooled_scores:
        pooled_pr_auc = precision_recall_auc(
            np.concatenate(pooled_scores),
            np.concatenate(pooled_labels),
            no_positive="zero",
        )
        assert pooled_pr_auc is not None
    else:
        pooled_pr_auc = 0.0

    return RetrievalMetrics(
        query_count=len(queries),
        evaluated_query_count=evaluated_count,
        skipped_query_count=len(queries) - evaluated_count,
        candidate_count=candidate_count,
        positive_count=positive_count,
        mrr=macro("reciprocal_rank"),
        hits_at=hits_at,
        recall_at=recall_at,
        micro_recall_at=micro_recall_at,
        question_pr_auc=macro("pr_auc"),
        candidate_pr_auc=pooled_pr_auc,
        per_query=per_query,
    )
