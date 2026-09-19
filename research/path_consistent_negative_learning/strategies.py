"""Matched supervision arms over a fixed set of candidates and features."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Iterable, Sequence

import networkx as nx

from .distances import (
    build_pair_distance_index,
    distances_to_any,
    makes_local_answer_progress,
    on_any_topic_answer_shortest_path,
)
from .schema import (
    DisputeCriterion,
    FixedCandidate,
    FixedCandidateBatch,
    StrategyAssignment,
    TopicAnswerPair,
    Transition,
)


STRATEGIES = (
    "strategy1_negative",
    "strategy2_ignore",
    "strategy3_positive",
    "random_drop",
)


def build_fixed_candidate_batch(
    *,
    graph: nx.Graph,
    query_id: str,
    transitions: Iterable[Transition],
    selected_positive_transitions: Iterable[Transition],
    topic_answer_pairs: Iterable[TopicAnswerPair],
    features: Sequence[Any] | None = None,
) -> FixedCandidateBatch:
    """Annotate policy-independent candidates for a single question.

    ``features`` is positional and is stored without copying, so callers can
    verify that every arm sees the same feature objects.  Candidate order is
    likewise preserved exactly.
    """

    fixed_transitions = tuple(tuple(item) for item in transitions)
    fixed_pairs = tuple(tuple(pair) for pair in topic_answer_pairs)
    positives = {tuple(item) for item in selected_positive_transitions}
    if features is None:
        fixed_features: tuple[Any, ...] = (None,) * len(fixed_transitions)
    else:
        fixed_features = tuple(features)
        if len(fixed_features) != len(fixed_transitions):
            raise ValueError("One feature object is required per candidate transition")

    pair_index = build_pair_distance_index(graph, fixed_pairs)
    answers = (answer for _, answer in fixed_pairs)
    answer_distance_index = distances_to_any(graph, answers)

    candidates = tuple(
        FixedCandidate(
            transition=transition,
            features=candidate_features,
            is_selected_positive=transition in positives,
            on_topic_answer_shortest_path=on_any_topic_answer_shortest_path(
                graph,
                transition,
                pair_index,
            ),
            local_answer_progress=makes_local_answer_progress(
                graph,
                transition,
                answer_distance_index,
            ),
        )
        for transition, candidate_features in zip(
            fixed_transitions,
            fixed_features,
            strict=True,
        )
    )
    return FixedCandidateBatch(
        query_id=query_id,
        candidates=candidates,
        topic_answer_pairs=fixed_pairs,
    )


def _baseline_labels(batch: FixedCandidateBatch) -> tuple[int, ...]:
    return tuple(int(candidate.is_selected_positive) for candidate in batch.candidates)


def strategy1_negative(batch: FixedCandidateBatch) -> StrategyAssignment:
    """Keep selected positives positive and label every other candidate negative."""

    size = len(batch.candidates)
    return StrategyAssignment(
        strategy="strategy1_negative",
        batch=batch,
        labels=_baseline_labels(batch),
        loss_mask=(True,) * size,
    )


def strategy2_ignore(
    batch: FixedCandidateBatch,
    *,
    criterion: DisputeCriterion = "pair_shortest_path",
) -> StrategyAssignment:
    """Mask disputed unselected candidates without resampling replacements."""

    return StrategyAssignment(
        strategy="strategy2_ignore",
        batch=batch,
        labels=_baseline_labels(batch),
        loss_mask=tuple(
            not candidate.is_disputed(criterion) for candidate in batch.candidates
        ),
        dispute_criterion=criterion,
    )


def strategy3_positive(
    batch: FixedCandidateBatch,
    *,
    criterion: DisputeCriterion = "pair_shortest_path",
) -> StrategyAssignment:
    """Promote every disputed candidate to a positive label."""

    return StrategyAssignment(
        strategy="strategy3_positive",
        batch=batch,
        labels=tuple(
            int(candidate.is_selected_positive or candidate.is_disputed(criterion))
            for candidate in batch.candidates
        ),
        loss_mask=(True,) * len(batch.candidates),
        dispute_criterion=criterion,
    )


def _stable_query_seed(seed: int, query_id: str) -> int:
    payload = f"{seed}\0{query_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def random_drop(
    batch: FixedCandidateBatch,
    *,
    seed: int,
    criterion: DisputeCriterion = "pair_shortest_path",
) -> StrategyAssignment:
    """Randomly mask as many baseline negatives as strategy 2 masks per question.

    The global seed is mixed with ``query_id`` using SHA-256.  Consequently the
    result is reproducible across Python processes and does not depend on the
    order in which questions are processed.
    """

    drop_count = sum(
        candidate.is_disputed(criterion) for candidate in batch.candidates
    )
    eligible_indices = [
        index
        for index, candidate in enumerate(batch.candidates)
        if not candidate.is_selected_positive
    ]
    rng = random.Random(_stable_query_seed(seed, batch.query_id))
    dropped_indices = set(rng.sample(eligible_indices, drop_count))
    return StrategyAssignment(
        strategy="random_drop",
        batch=batch,
        labels=_baseline_labels(batch),
        loss_mask=tuple(
            index not in dropped_indices for index in range(len(batch.candidates))
        ),
        dispute_criterion=criterion,
    )


def apply_strategy(
    batch: FixedCandidateBatch,
    strategy: str,
    *,
    seed: int | None = None,
    criterion: DisputeCriterion = "pair_shortest_path",
) -> StrategyAssignment:
    """Apply a named arm while keeping the fixed batch unchanged."""

    if strategy == "strategy1_negative":
        return strategy1_negative(batch)
    if strategy == "strategy2_ignore":
        return strategy2_ignore(batch, criterion=criterion)
    if strategy == "strategy3_positive":
        return strategy3_positive(batch, criterion=criterion)
    if strategy == "random_drop":
        if seed is None:
            raise ValueError("random_drop requires an explicit seed")
        return random_drop(batch, seed=seed, criterion=criterion)
    raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")
