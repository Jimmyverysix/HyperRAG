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
