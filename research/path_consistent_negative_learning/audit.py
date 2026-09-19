"""Audit path-consistent negatives in a generated HyperRAG training file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx

from .core import answer_distances, filter_path_consistent_negatives


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit_samples(
    graph: nx.Graph,
    samples: list[dict[str, Any]],
    answer_map: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    per_query = []
    total_negatives = 0
    total_path_consistent = 0

    for sample in samples:
        query = sample["query"]
        answers = sample.get("answer_entities")
        if answers is None and answer_map is not None:
            answers = answer_map.get(query)
        if answers is None:
            raise ValueError(
                f"No answer_entities for query {query!r}; regenerate samples with the "
                "patched prepare.py or pass --answer-map."
            )

        negatives = sample.get("negative_triplets", [])
        distances = answer_distances(graph, answers)
        _, filtered = filter_path_consistent_negatives(negatives, distances)
        count = len(negatives)
        filtered_count = len(filtered)
        total_negatives += count
        total_path_consistent += filtered_count
        per_query.append(
            {
                "query": query,
                "negative_count": count,
                "path_consistent_negative_count": filtered_count,
                "path_consistent_negative_ratio": (
                    filtered_count / count if count else 0.0
                ),
            }
        )

    return {
        "sample_count": len(samples),
        "negative_count": total_negatives,
        "path_consistent_negative_count": total_path_consistent,
        "path_consistent_negative_ratio": (
            total_path_consistent / total_negatives if total_negatives else 0.0
        ),
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit sampled HyperRAG negatives that make optimal progress to an answer."
    )
    parser.add_argument("--graph", type=Path, required=True, help="HyperRAG GraphML file")
    parser.add_argument("--samples", type=Path, required=True, help="retrieval_samples JSON")
    parser.add_argument(
        "--answer-map",
        type=Path,
        help="Optional JSON object mapping query strings to graph answer-node lists",
    )
    parser.add_argument("--output", type=Path, required=True, help="Audit JSON output")
    args = parser.parse_args()

    graph = nx.read_graphml(args.graph)
    samples = _load_json(args.samples)
    answer_map = _load_json(args.answer_map) if args.answer_map else None
    report = audit_samples(graph, samples, answer_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "per_query"}, indent=2))


if __name__ == "__main__":
    main()

