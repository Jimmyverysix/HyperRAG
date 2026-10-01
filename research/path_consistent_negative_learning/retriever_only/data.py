"""Load and align structured WikiTopics queries with released NLG text."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping

from .alignment import align_queries, text_mentions_topic
from .labels import relation_base_id


QUERY_SHAPE = ("e", ("r", "r", "r"))
NLG_QUERY_SHAPE = "('e', ('r', 'r', 'r'))"


def _trusted_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - public benchmark format


def _shape_payload(container: object, shape: object, label: str) -> object:
    if not isinstance(container, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    if shape in container:
        return container[shape]
    raise ValueError(f"{label} has no {shape!r} query family")


@dataclass(frozen=True)
class LabelSnapshot:
    """Frozen English labels used by the released KG/NLG conversion."""

    labels: Mapping[str, str]
    source: str
    retrieved_at: str

    @classmethod
    def load(cls, path: Path) -> "LabelSnapshot":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported label snapshot: {path}")
        labels = payload.get("labels")
        if not isinstance(labels, Mapping):
            raise ValueError(f"Label snapshot has no labels mapping: {path}")
        return cls(
            labels={str(key): str(value) for key, value in labels.items()},
            source=str(payload["source"]),
            retrieved_at=str(payload["retrieved_at"]),
        )

    def contains_relation(self, relation_id: str) -> bool:
        return relation_base_id(relation_id) in self.labels


@dataclass(frozen=True)
class DomainMappings:
    train_entities: Mapping[int, str]
    test_entities: Mapping[int, str]
    relations: Mapping[int, str]

    @classmethod
    def load(cls, path: Path) -> "DomainMappings":
        payload = _trusted_pickle(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Expected mappings in {path}")

        def reverse(name: str) -> dict[int, str]:
            value = payload.get(name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path} is missing {name}")
            return {int(index): str(identifier) for identifier, index in value.items()}

        return cls(
            train_entities=reverse("e2id_train"),
            test_entities=reverse("e2id_test"),
            relations=reverse("r2id"),
        )

    def entity_map(self, split: str) -> Mapping[int, str]:
        return self.test_entities if split == "test" else self.train_entities


@dataclass(frozen=True)
class AlignedQuery:
    """One structured query paired with its released natural-language text."""

    key: str
    domain: str
    split: str
    raw_topic_id: int
    raw_relation_ids: tuple[int, int, int]
    topic_qid: str
    relation_qids: tuple[str, str, str]
    answer_qids: tuple[str, ...]
    topic_node: str | None
    relation_texts: tuple[str, str, str]
    answer_nodes: tuple[str, ...]
    text: str
    topic_text_alignment_evidence: bool
    alignment_supported: bool


def _query_files(split: str) -> tuple[str, str]:
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return f"{split}_queries.pkl", f"{split}_answers_hard.pkl"


def _iter_raw_queries(payload: object) -> Iterable[object]:
    if isinstance(payload, Mapping):
        return payload.keys()
    if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        return payload
    raise ValueError("Structured query payload must be iterable")


def _parse_query(raw: object) -> tuple[int, tuple[int, int, int]]:
    try:
        topic, relations = raw  # type: ignore[misc]
        relation_ids = tuple(int(value) for value in relations)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Malformed structured query: {raw!r}") from exc
    if len(relation_ids) != 3:
        raise ValueError(f"Expected a three-hop query: {raw!r}")
    return int(topic), relation_ids  # type: ignore[return-value]


def _query_key(
    domain: str,
    split: str,
    topic_id: int,
    relation_ids: tuple[int, int, int],
) -> str:
    relation_part = ",".join(str(value) for value in relation_ids)
    return f"{domain}:{split}:{topic_id}:{relation_part}"


def load_aligned_queries(
    structured_domain_dir: Path,
    nlg_domain_dir: Path,
    split: str,
    labels: LabelSnapshot,
) -> tuple[AlignedQuery, ...]:
    """Replay released label filtering, then zip structured and NLG queries."""

    query_name, answer_name = _query_files(split)
    raw_queries = _shape_payload(
        _trusted_pickle(structured_domain_dir / query_name),
        QUERY_SHAPE,
        query_name,
    )
    raw_answers = _shape_payload(
        _trusted_pickle(structured_domain_dir / answer_name),
        QUERY_SHAPE,
        answer_name,
    )
    if not isinstance(raw_answers, Mapping):
        raise ValueError(f"{answer_name} must map queries to answer IDs")
    nlg_payload = json.loads(
        (nlg_domain_dir / f"{split}_queries.json").read_text(encoding="utf-8")
    )
    nlg_queries = nlg_payload.get(NLG_QUERY_SHAPE)
    if not isinstance(nlg_queries, list) or not all(
        isinstance(value, str) for value in nlg_queries
    ):
        raise ValueError(f"Malformed NLG queries for {structured_domain_dir.name}/{split}")

    mappings = DomainMappings.load(structured_domain_dir / "og_mappings.pkl")
    entities = mappings.entity_map(split)
    answer_lookup = {raw: values for raw, values in raw_answers.items()}
    raw_records: list[
        tuple[object, int, tuple[int, int, int], str | None, tuple[str, str, str]]
    ] = []
    for raw_query in _iter_raw_queries(raw_queries):
        topic_id, relation_ids = _parse_query(raw_query)
        topic_qid = entities.get(topic_id)
        relation_qids = tuple(mappings.relations.get(value, "") for value in relation_ids)
        if not all(value and labels.contains_relation(value) for value in relation_qids):
            raise ValueError(f"Missing relation label for structured query {raw_query!r}")
        raw_records.append(
            (
                raw_query,
                topic_id,
                relation_ids,
                topic_qid,
                relation_qids,  # type: ignore[arg-type]
            )
        )
        raw_answer_ids = answer_lookup.get(raw_query)
        if raw_answer_ids is None:
            raise ValueError(f"Missing answers for {raw_query!r}")

    pairs = align_queries(
        [
            labels.labels.get(topic_qid) if topic_qid is not None else None
            for _, _, _, topic_qid, _ in raw_records
        ],
        nlg_queries,
    )
    aligned: list[AlignedQuery] = []
    for pair in pairs:
        raw_query, topic_id, relation_ids, topic_qid, relation_qids = raw_records[
            pair.structured_index
        ]
        text = nlg_queries[pair.nlg_index]
        if topic_qid is None:
            raise ValueError("Structured topic ID is absent from the domain mapping")
        answer_ids = answer_lookup.get(raw_query)
        if answer_ids is None:
            raise ValueError(f"Missing answers for {raw_query!r}")
        answer_qids = tuple(
            sorted(
                {
                    entities[int(answer_id)]
                    for answer_id in answer_ids
                    if int(answer_id) in entities
                }
            )
        )
        relation_texts = tuple(
            labels.labels[relation_base_id(relation_qid)]
            + (" (inverse)" if relation_qid.endswith("_inv") else "")
            for relation_qid in relation_qids
        )
        topic_label = labels.labels.get(topic_qid)
        topic_node = topic_qid if topic_label is not None else None
        answer_nodes = tuple(
            sorted(
                {
                    answer_qid
                    for answer_qid in answer_qids
                    if answer_qid in labels.labels
                }
            )
        )
        topic_evidence = topic_label is not None and text_mentions_topic(
            text, topic_label
        )
        aligned.append(
            AlignedQuery(
                key=_query_key(
                    structured_domain_dir.name,
                    split,
                    topic_id,
                    relation_ids,
                ),
                domain=structured_domain_dir.name,
                split=split,
                raw_topic_id=topic_id,
                raw_relation_ids=relation_ids,
                topic_qid=topic_qid,
                relation_qids=relation_qids,
                answer_qids=answer_qids,
                topic_node=topic_node,
                relation_texts=relation_texts,  # type: ignore[arg-type]
                answer_nodes=answer_nodes,
                text=text,
                topic_text_alignment_evidence=topic_evidence,
                # The monotonic release-order alignment is answer-free.  A
                # current topic label need not be repeated verbatim by the NLG
                # question, so exact mention is diagnostic rather than a gate.
                alignment_supported=topic_node is not None,
            )
        )
    return tuple(aligned)
