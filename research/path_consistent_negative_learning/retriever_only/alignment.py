"""Monotonic alignment of released NLG questions to structured queries."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Sequence


NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_like = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(NON_ALPHANUMERIC.sub(" ", ascii_like).split())


def text_mentions_topic(text: str, topic: str) -> bool:
    normalized_topic = normalize_text(topic)
    return bool(normalized_topic) and normalized_topic in normalize_text(text)


def _topic_cost(normalized_topic: str | None, normalized_question: str) -> float:
    if normalized_topic is None:
        # Some labels available during the released conversion have since been
        # removed from Wikidata.  Exact neighboring anchors still determine the
        # monotonic alignment; these rows are retained only to preserve order and
        # are later excluded from graph-based training/evaluation.
        return 1.5
    if normalized_topic and normalized_topic in normalized_question:
        return 0.0
    topic_tokens = normalized_topic.split()
    question_tokens = normalized_question.split()
    if topic_tokens and all(token in question_tokens for token in topic_tokens):
        return 0.25
    return 2.0


@dataclass(frozen=True)
class AlignmentPair:
    structured_index: int
    nlg_index: int


def align_queries(
    topics: Sequence[str | None],
    nlg_questions: Sequence[str],
) -> tuple[AlignmentPair, ...]:
    """Align ordered sequences without consulting answer annotations.

    The released conversion preserves order and only filters structured queries.
    Dynamic programming therefore needs match and structured-delete transitions,
    with a narrow state dimension equal to the observed length difference.
    """

    if len(topics) < len(nlg_questions):
        raise ValueError("NLG contains more queries than the structured source")
    deletions = len(topics) - len(nlg_questions)
    normalized_topics = [
        normalize_text(topic) if topic is not None else None for topic in topics
    ]
    normalized_questions = [normalize_text(value) for value in nlg_questions]
    infinity = math.inf
    previous = [infinity] * (deletions + 1)
    previous[0] = 0.0
    parents: list[list[str | None]] = [
        [None] * (deletions + 1) for _ in range(len(topics) + 1)
    ]

    for index, topic in enumerate(normalized_topics):
        current = [infinity] * (deletions + 1)
        for deleted in range(min(index, deletions) + 1):
            value = previous[deleted]
            if not math.isfinite(value):
                continue
            nlg_index = index - deleted
            if nlg_index < len(nlg_questions):
                cost = _topic_cost(topic, normalized_questions[nlg_index])
                candidate = value + cost
                if candidate <= current[deleted]:
                    current[deleted] = candidate
                    parents[index + 1][deleted] = "match"
            if deleted < deletions and value < current[deleted + 1]:
                current[deleted + 1] = value
                parents[index + 1][deleted + 1] = "delete"
        previous = current

    if not math.isfinite(previous[deletions]):
        raise ValueError("No valid monotonic query alignment")
    pairs: list[AlignmentPair] = []
    index = len(topics)
    deleted = deletions
    while index:
        action = parents[index][deleted]
        if action == "delete":
            deleted -= 1
            index -= 1
            continue
        if action != "match":
            raise RuntimeError("Incomplete alignment backtrace")
        structured_index = index - 1
        nlg_index = structured_index - deleted
        pairs.append(AlignmentPair(structured_index, nlg_index))
        index -= 1
    pairs.reverse()
    if len(pairs) != len(nlg_questions):
        raise RuntimeError("Alignment did not consume every NLG question")
    return tuple(pairs)
