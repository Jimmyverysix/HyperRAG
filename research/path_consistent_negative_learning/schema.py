"""Shared, immutable data structures for supervision-policy experiments.

The candidate batch is deliberately separated from its supervision assignment.
This makes it difficult for a policy to accidentally change candidate order or
model features while changing labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Literal, TypeAlias


Node: TypeAlias = Hashable
Transition: TypeAlias = tuple[Node, Node, Node]
TopicAnswerPair: TypeAlias = tuple[Node, Node]
DisputeCriterion: TypeAlias = Literal[
    "pair_shortest_path",
    "local_answer_progress",
]


@dataclass(frozen=True)
class FixedCandidate:
    """One candidate and the policy-independent information attached to it."""

    transition: Transition
    features: Any
    is_selected_positive: bool
    on_topic_answer_shortest_path: bool
    local_answer_progress: bool

    def is_disputed(
        self,
        criterion: DisputeCriterion = "pair_shortest_path",
    ) -> bool:
        """Return whether an unselected candidate meets an ambiguity rule."""

        if self.is_selected_positive:
            return False
        if criterion == "pair_shortest_path":
            return self.on_topic_answer_shortest_path
        if criterion == "local_answer_progress":
            return self.local_answer_progress
        raise ValueError(f"Unknown dispute criterion: {criterion!r}")


@dataclass(frozen=True)
class FixedCandidateBatch:
    """Policy-independent candidates for exactly one question."""

    query_id: str
    candidates: tuple[FixedCandidate, ...]
    topic_answer_pairs: tuple[TopicAnswerPair, ...]

    def __post_init__(self) -> None:
        transitions = [candidate.transition for candidate in self.candidates]
        if len(transitions) != len(set(transitions)):
            raise ValueError("Candidate transitions must be unique within a question")


@dataclass(frozen=True)
class StrategyAssignment:
    """Labels, masks, and optional weights for one fixed candidate batch."""

    strategy: str
    batch: FixedCandidateBatch
    labels: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    dispute_criterion: DisputeCriterion = "pair_shortest_path"
    loss_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        size = len(self.batch.candidates)
        if len(self.labels) != size or len(self.loss_mask) != size:
            raise ValueError("Labels and loss masks must align with the candidate batch")
        if any(label not in (0, 1) for label in self.labels):
            raise ValueError("Binary supervision labels must be either 0 or 1")
        if self.loss_weights is None:
            object.__setattr__(
                self,
                "loss_weights",
                tuple(float(active) for active in self.loss_mask),
            )
        elif len(self.loss_weights) != size:
            raise ValueError("Loss weights must align with the candidate batch")
        elif any(weight < 0.0 for weight in self.loss_weights):
            raise ValueError("Loss weights must be non-negative")
        if any(active != (weight > 0.0) for active, weight in zip(
            self.loss_mask,
            self.loss_weights,
            strict=True,
        )):
            raise ValueError("Loss masks must identify exactly the positive loss weights")

    @property
    def active_count(self) -> int:
        """Number of candidates contributing to the loss."""

        return sum(self.loss_mask)

    @property
    def ignored_count(self) -> int:
        """Number of candidates excluded from the loss."""

        return len(self.loss_mask) - self.active_count

    @property
    def effective_weight(self) -> float:
        """Total supervision weight contributed by the batch."""

        assert self.loss_weights is not None
        return sum(self.loss_weights)
