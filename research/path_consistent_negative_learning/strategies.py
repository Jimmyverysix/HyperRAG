"""Matched supervision arms over a fixed set of candidates and features."""

from __future__ import annotations

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

WEIGHTED_STRATEGIES = (
    "strategy2_weighted",
    "random_weighted",
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

    return strategy2_weighted(
        batch,
        disputed_negative_weight=0.0,
        criterion=criterion,
        strategy_name="strategy2_ignore",
    )


def strategy2_weighted(
    batch: FixedCandidateBatch,
    *,
    disputed_negative_weight: float,
    criterion: DisputeCriterion = "pair_shortest_path",
    strategy_name: str = "strategy2_weighted",
) -> StrategyAssignment:
    """Downweight disputed negatives while leaving their labels unchanged."""

    if not 0.0 <= disputed_negative_weight <= 1.0:
        raise ValueError("disputed_negative_weight must be between 0 and 1")
    weights = tuple(
        disputed_negative_weight if candidate.is_disputed(criterion) else 1.0
        for candidate in batch.candidates
    )
    return StrategyAssignment(
        strategy=strategy_name,
        batch=batch,
        labels=_baseline_labels(batch),
        loss_mask=tuple(weight > 0.0 for weight in weights),
        dispute_criterion=criterion,
        loss_weights=weights,
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


def random_drop(
    batch: FixedCandidateBatch,
    *,
    seed: int,
    criterion: DisputeCriterion = "pair_shortest_path",
) -> StrategyAssignment:
    """Randomly mask as many baseline negatives as strategy 2 masks per question.

    The global seed and ``query_id`` seed a dedicated standard-library random
    generator.  Consequently the result is reproducible across Python
    processes and does not depend on question-processing order.
    """

    return random_weighted(
        batch,
        seed=seed,
        disputed_negative_weight=0.0,
        criterion=criterion,
        strategy_name="random_drop",
    )


def random_weighted(
    batch: FixedCandidateBatch,
    *,
    seed: int,
    disputed_negative_weight: float,
    criterion: DisputeCriterion = "pair_shortest_path",
    strategy_name: str = "random_weighted",
) -> StrategyAssignment:
    """Downweight a matched random set of baseline negatives per question."""

    if not 0.0 <= disputed_negative_weight <= 1.0:
        raise ValueError("disputed_negative_weight must be between 0 and 1")
    weighted_count = sum(
        candidate.is_disputed(criterion) for candidate in batch.candidates
    )
    eligible_indices = [
        index
        for index, candidate in enumerate(batch.candidates)
        if not candidate.is_selected_positive
    ]
    rng = random.Random()
    rng.seed(f"{seed}\0{batch.query_id}", version=2)
    weighted_indices = set(rng.sample(eligible_indices, weighted_count))
    weights = tuple(
        disputed_negative_weight if index in weighted_indices else 1.0
        for index in range(len(batch.candidates))
    )
    return StrategyAssignment(
        strategy=strategy_name,
        batch=batch,
        labels=_baseline_labels(batch),
        loss_mask=tuple(weight > 0.0 for weight in weights),
        dispute_criterion=criterion,
        loss_weights=weights,
    )


def apply_strategy(
    batch: FixedCandidateBatch,
    strategy: str,
    *,
    seed: int | None = None,
    disputed_negative_weight: float | None = None,
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
    if strategy == "strategy2_weighted":
        if disputed_negative_weight is None:
            raise ValueError("strategy2_weighted requires disputed_negative_weight")
        return strategy2_weighted(
            batch,
            disputed_negative_weight=disputed_negative_weight,
            criterion=criterion,
        )
    if strategy == "random_weighted":
        if seed is None:
            raise ValueError("random_weighted requires an explicit seed")
        if disputed_negative_weight is None:
            raise ValueError("random_weighted requires disputed_negative_weight")
        return random_weighted(
            batch,
            seed=seed,
            disputed_negative_weight=disputed_negative_weight,
            criterion=criterion,
        )
    choices = STRATEGIES + WEIGHTED_STRATEGIES
    raise ValueError(f"Unknown strategy {strategy!r}; choose from {choices}")
