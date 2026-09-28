"""Validate all WikiTopics/NLG alignments before formal GPU preparation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Sequence

from .alignment import normalize_text, text_mentions_topic
from .data import NLG_QUERY_SHAPE, LabelSnapshot, load_aligned_queries
from .graph import build_deterministic_hypergraph
from .provenance import collect_provenance


REPOSITORY = Path(__file__).resolve().parents[3]


def validate_domain(
    structured_root: Path,
    nlg_root: Path,
    domain: str,
    labels: LabelSnapshot,
) -> dict[str, object]:
    """Validate one domain and summarize evaluation eligibility."""

    structured_dir = structured_root / domain
    nlg_dir = nlg_root / domain
    bundle = build_deterministic_hypergraph(structured_dir, labels)
    split_reports: dict[str, object] = {}
    for split in ("train", "valid", "test"):
        queries = load_aligned_queries(structured_dir, nlg_dir, split, labels)
        released_answers_payload = json.loads(
            (nlg_dir / f"{split}_answers_hard.json").read_text(encoding="utf-8")
        )
        released_answers = released_answers_payload[NLG_QUERY_SHAPE]
        labeled_topics = sum(query.topic_node is not None for query in queries)
        topic_mentions = sum(
            query.topic_node is not None
            and text_mentions_topic(query.text, labels.labels[query.topic_node])
            for query in queries
        )
        topic_in_graph = sum(query.topic_node in bundle.graph for query in queries)
        answer_in_graph = sum(
            any(answer in bundle.graph for answer in query.answer_nodes)
            for query in queries
        )
        eligible = sum(
            query.alignment_supported
            and query.topic_node in bundle.graph
            and any(answer in bundle.graph for answer in query.answer_nodes)
            for query in queries
        )
        comparable_answers = 0
        overlapping_answers = 0
        question_counts = Counter(query.text for query in queries)
        for query in queries:
            # The released conversion stores answers in a dict keyed by generated
            # question text, so duplicate questions overwrite earlier answers.
            # Those rows remain valid queries but cannot audit alignment by the
            # overwritten answer value.
            if question_counts[query.text] != 1:
                continue
            structured = {
                normalize_text(labels.labels[value]) for value in query.answer_nodes
            }
            released = {
                normalize_text(str(value))
                for value in released_answers.get(query.text, [])
            }
            if structured and released:
                comparable_answers += 1
                overlapping_answers += bool(structured & released)
        topic_mention_rate = topic_mentions / labeled_topics if labeled_topics else 0.0
        answer_overlap_rate = (
            overlapping_answers / comparable_answers if comparable_answers else 0.0
        )
        split_reports[split] = {
            "aligned_queries": len(queries),
            "queries_with_current_topic_label": labeled_topics,
            "question_mentions_current_topic_label": topic_mentions,
            "topic_mention_rate": topic_mention_rate,
            "duplicate_question_keys": sum(
                count > 1 for count in question_counts.values()
            ),
            "duplicate_question_occurrences": sum(
                count for count in question_counts.values() if count > 1
            ),
            "comparable_hard_answer_queries": comparable_answers,
            "hard_answer_label_overlap_queries": overlapping_answers,
            "hard_answer_label_overlap_rate": answer_overlap_rate,
            "alignment_supported_queries": sum(
                query.alignment_supported for query in queries
            ),
            "alignment_unsupported_queries": sum(
                not query.alignment_supported for query in queries
            ),
            "topic_in_graph": topic_in_graph,
            "at_least_one_answer_in_graph": answer_in_graph,
            "retriever_eligible_queries": eligible,
            "excluded_queries": len(queries) - eligible,
        }
    entity_labels = [
        bundle.node_texts[node]
        for node, data in bundle.graph.nodes(data=True)
        if data["kind"] == "entity"
    ]
    label_counts = Counter(entity_labels)
    return {
        "domain": domain,
        "entity_identity": "wikidata_qid",
        "graph_nodes": bundle.graph.number_of_nodes(),
        "graph_edges": bundle.graph.number_of_edges(),
        "entity_nodes": sum(
            1 for _, data in bundle.graph.nodes(data=True) if data["kind"] == "entity"
        ),
        "hyperedge_nodes": sum(
            1
            for _, data in bundle.graph.nodes(data=True)
            if data["kind"] == "hyperedge"
        ),
        "duplicate_entity_label_groups": sum(
            count > 1 for count in label_counts.values()
        ),
        "entities_with_duplicate_labels": sum(
            count for count in label_counts.values() if count > 1
        ),
        "fact_count": bundle.fact_count,
        "train_structured_groups": bundle.train_group_count,
        "test_structured_groups": bundle.test_group_count,
        "splits": split_reports,
    }


def validate_all(
    structured_root: Path,
    nlg_root: Path,
    label_snapshot: Path,
    domains: Sequence[str],
) -> dict[str, object]:
    labels = LabelSnapshot.load(label_snapshot)
    reports = [
        validate_domain(structured_root, nlg_root, domain, labels)
        for domain in domains
    ]
    return {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "zero_llm_deterministic_wikitopics_alignment",
        "structured_root": str(structured_root.resolve()),
        "nlg_root": str(nlg_root.resolve()),
        "label_snapshot": str(label_snapshot.resolve()),
        "label_source": labels.source,
        "label_retrieved_at": labels.retrieved_at,
        "domains": reports,
        "domain_count": len(reports),
        "status": "passed",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-root", type=Path, required=True)
    parser.add_argument("--nlg-root", type=Path, required=True)
    parser.add_argument("--label-snapshot", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_all(
        args.structured_root,
        args.nlg_root,
        args.label_snapshot,
        args.domains,
    )
    report["provenance"] = collect_provenance(REPOSITORY, sys.argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"data validation passed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
