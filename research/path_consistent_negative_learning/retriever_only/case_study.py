"""Extract one deterministic, real path-supervision conflict example."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import networkx as nx

from .candidates import build_training_query
from .data import LabelSnapshot, load_aligned_queries
from .graph import (
    DeterministicHypergraph,
    Transition,
    all_shortest_distances,
    build_deterministic_hypergraph,
    selected_paths,
)


def witness_shortest_path(
    graph: nx.Graph,
    topic: str,
    answers: Iterable[str],
    transition: Transition,
) -> tuple[str, ...] | None:
    """Return a shortest topic--answer path containing ``transition``."""

    head, edge, tail = transition
    source_distances = all_shortest_distances(graph, topic)
    if head not in source_distances:
        return None
    for answer in sorted(set(answers)):
        if answer not in source_distances:
            continue
        try:
            tail_distance = nx.shortest_path_length(graph, tail, answer)
        except nx.NetworkXNoPath:
            continue
        if source_distances[head] + 2 + tail_distance != source_distances[answer]:
            continue
        prefix = nx.shortest_path(graph, topic, head)
        suffix = nx.shortest_path(graph, tail, answer)
        return tuple([*prefix, edge, tail, *suffix[1:]])
    return None


def _render_path(
    path: Iterable[str],
    node_texts: Mapping[str, str],
) -> list[dict[str, str]]:
    return [
        {
            "node_id": node,
            "kind": "hyperedge" if node.startswith("H:") else "entity",
            "text": node_texts[node],
        }
        for node in path
    ]


def extract_case_study(
    structured_root: Path,
    nlg_root: Path,
    label_snapshot: Path,
    *,
    domain: str,
    seed: int,
) -> dict[str, Any]:
    """Select the first released-order training query with a sampled conflict."""

    labels = LabelSnapshot.load(label_snapshot)
    structured_dir = structured_root / domain
    queries = load_aligned_queries(
        structured_dir,
        nlg_root / domain,
        "train",
        labels,
    )
    bundle: DeterministicHypergraph = build_deterministic_hypergraph(
        structured_dir,
        labels,
    )
    rng = random.Random(seed)
    for aligned in queries:
        prepared = build_training_query(aligned, bundle.graph, rng=rng)
        if prepared is None or not prepared.path_consistent_negatives:
            continue
        conflict = prepared.path_consistent_negatives[0]
        witness = witness_shortest_path(
            bundle.graph,
            prepared.topic,
            prepared.answers,
            conflict,
        )
        if witness is None:
            raise RuntimeError("path-consistent transition has no witness path")
        selected = selected_paths(
            bundle.graph,
            prepared.topic,
            prepared.answers,
        )
        source_distances = all_shortest_distances(bundle.graph, prepared.topic)
        answer = witness[-1]
        return {
            "schema_version": 1,
            "selection_rule": "first_released_order_query_with_sampled_conflict",
            "domain": domain,
            "seed": seed,
            "query_key": prepared.key,
            "question": prepared.text,
            "topic": {
                "node_id": prepared.topic,
                "text": bundle.node_texts[prepared.topic],
            },
            "answers": [
                {"node_id": value, "text": bundle.node_texts[value]}
                for value in prepared.answers
            ],
            "sampled_negative_transition": {
                "head": conflict[0],
                "hyperedge": conflict[1],
                "tail": conflict[2],
                "head_text": bundle.node_texts[conflict[0]],
                "hyperedge_text": bundle.node_texts[conflict[1]],
                "tail_text": bundle.node_texts[conflict[2]],
            },
            "distance_equality": {
                "topic_to_head": source_distances[conflict[0]],
                "transition_cost": 2,
                "tail_to_answer": nx.shortest_path_length(
                    bundle.graph, conflict[2], answer
                ),
                "topic_to_answer": source_distances[answer],
            },
            "selected_paths": [
                _render_path(path, bundle.node_texts) for path in selected
            ],
            "conflict_witness_path": _render_path(witness, bundle.node_texts),
        }
    raise ValueError(f"no sampled path-consistent negative found in {domain}/seed={seed}")
