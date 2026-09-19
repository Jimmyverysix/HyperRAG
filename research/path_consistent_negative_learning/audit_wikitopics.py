"""CLI and reporting layer for the raw WikiTopics KG prevalence audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .wikitopics import (
    THREE_HOP_SHAPE,
    QueryExample,
    Transition,
    WikiTopicsDomain,
    build_query_graph,
    candidate_pool,
    disputed_shortest_path_transitions,
    iter_mapping_sizes,
    load_domain,
    simulate_original_sampler,
    transition_arity,
    transition_depth,
    transition_dict,
)


STRATIFICATION_DIMENSIONS = ("hop", "arity", "depth")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _seed_values(raw_values: Iterable[int]) -> tuple[int, ...]:
    seeds = tuple(sorted(set(int(value) for value in raw_values)))
    if not seeds:
        raise ValueError("At least one sampling seed is required")
    return seeds


def _dimension_values(raw_values: Iterable[str]) -> tuple[str, ...]:
    dimensions = tuple(dict.fromkeys(raw_values))
    unsupported = sorted(set(dimensions) - set(STRATIFICATION_DIMENSIONS))
    if unsupported:
        raise ValueError(f"Unsupported strata: {unsupported}")
    return dimensions


def _stratum_value(
    dimension: str,
    transition: Transition,
    domain: WikiTopicsDomain,
    source_distances: Mapping[tuple[str, int], int],
    query_hop: int | None,
) -> int | None:
    if dimension == "hop":
        return query_hop
    if dimension == "arity":
        return transition_arity(domain.graph, transition)
    if dimension == "depth":
        return transition_depth(source_distances, transition)
    raise ValueError(f"Unknown stratification dimension: {dimension!r}")


def _per_query_strata(
    dimensions: Sequence[str],
    domain: WikiTopicsDomain,
    candidates: Sequence[Transition],
    disputed: set[Transition],
    samples_by_seed: Mapping[int, Sequence[Transition]],
    source_distances: Mapping[tuple[str, int], int],
    query_hop: int | None,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for dimension in dimensions:
        pool: dict[int | None, list[int]] = defaultdict(lambda: [0, 0])
        sampled: dict[int | None, dict[int, list[int]]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0])
        )
        for transition in candidates:
            value = _stratum_value(
                dimension, transition, domain, source_distances, query_hop
            )
            pool[value][0] += 1
            pool[value][1] += int(transition in disputed)
        for seed, transitions in samples_by_seed.items():
            for transition in transitions:
                value = _stratum_value(
                    dimension, transition, domain, source_distances, query_hop
                )
                sampled[value][seed][0] += 1
                sampled[value][seed][1] += int(transition in disputed)

        records: list[dict[str, Any]] = []
        for value in sorted(pool, key=lambda item: (-1 if item is None else item)):
            candidate_count, disputed_count = pool[value]
            records.append(
                {
                    "value": value,
                    "candidate_pool_size": candidate_count,
                    "disputed_candidate_count": disputed_count,
                    "disputed_candidate_rate": _ratio(disputed_count, candidate_count),
                    "sampling_by_seed": [
                        {
                            "seed": seed,
                            "sampled_count": sampled[value][seed][0],
                            "disputed_sampled_count": sampled[value][seed][1],
                            "disputed_sampled_rate": _ratio(
                                sampled[value][seed][1], sampled[value][seed][0]
                            ),
                        }
                        for seed in sorted(samples_by_seed)
                    ],
                }
            )
        output[dimension] = records
    return output


def audit_example(
    domain: WikiTopicsDomain,
    example: QueryExample,
    query_index: int,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    stratify: Sequence[str] = STRATIFICATION_DIMENSIONS,
) -> dict[str, Any]:
    """Audit one structured query against a shared deterministic candidate pool."""

    seeds = _seed_values(seeds)
    stratify = _dimension_values(stratify)

    query_graph = build_query_graph(
        domain.graph, example.query.topic_id, example.answer_ids
    )
    candidates = candidate_pool(
        query_graph.subgraph, query_graph.selected_positives
    )
    disputed_transitions = disputed_shortest_path_transitions(
        domain.graph,
        example.query.topic_id,
        query_graph.reachable_answers,
        candidates,
        source_distances=query_graph.source_distances,
    )
    disputed = set(disputed_transitions)
    source_distances = query_graph.source_distances

    requested_samples = len(query_graph.selected_positives)
    samples_by_seed = {
        seed: simulate_original_sampler(
            query_graph.subgraph,
            query_graph.selected_positives,
            requested_samples,
            seed,
        )
        for seed in seeds
    }
    sampling_records = []
    for seed in seeds:
        samples = samples_by_seed[seed]
        disputed_samples = tuple(value for value in samples if value in disputed)
        sampling_records.append(
            {
                "seed": seed,
                "requested_sample_count": requested_samples,
                "sampled_count": len(samples),
                "disputed_sampled_count": len(disputed_samples),
                "disputed_sampled_rate": _ratio(len(disputed_samples), len(samples)),
                "disputed_samples": [
                    transition_dict(
                        transition,
                        domain.graph,
                        transition_depth(source_distances, transition),
                    )
                    for transition in disputed_samples
                ],
            }
        )

    path_hops = [int((len(path) - 1) / 2) for path in query_graph.selected_paths]
    status = "audited" if query_graph.reachable_answers else "no_reachable_answer"
    return {
        "schema_version": 1,
        "domain": domain.name,
        "query_index": query_index,
        "query_shape": ["e", ["r", "r", "r"]],
        "query": example.query.to_dict(),
        "answer_ids": list(example.answer_ids),
        "reachable_answer_ids": list(query_graph.reachable_answers),
        "missing_answer_ids": list(query_graph.missing_answers),
        "status": status,
        "selected_path_hops": path_hops,
        "min_selected_hop": query_graph.min_hop,
        "max_selected_hop": query_graph.max_hop,
        "selected_positive_count": len(query_graph.selected_positives),
        "selected_positive_transitions": [
            transition_dict(
                transition,
                domain.graph,
                transition_depth(source_distances, transition),
            )
            for transition in query_graph.selected_positives
        ],
        "candidate_pool_policy": "shared_across_label_strategies",
        "candidate_pool_size": len(candidates),
        "disputed_candidate_count": len(disputed_transitions),
        "disputed_candidate_rate": _ratio(
            len(disputed_transitions), len(candidates)
        ),
        "disputed_definition": "unselected transition on any full-graph topic--hard-answer shortest path",
        "disputed_transitions": [
            transition_dict(
                transition,
                domain.graph,
                transition_depth(source_distances, transition),
            )
            for transition in disputed_transitions
        ],
        "sampling": sampling_records,
        "strata": _per_query_strata(
            stratify,
            domain,
            candidates,
            disputed,
            samples_by_seed,
            source_distances,
            query_graph.max_hop,
        ),
    }


class SummaryAccumulator:
    """Streaming accumulator so a real-domain audit need not retain JSONL rows."""

    def __init__(
        self,
        domain: WikiTopicsDomain,
        seeds: Sequence[int],
        stratify: Sequence[str],
    ) -> None:
        self.domain = domain
        self.seeds = _seed_values(seeds)
        self.stratify = _dimension_values(stratify)
        self.query_count = 0
        self.audited_query_count = 0
        self.no_reachable_answer_count = 0
        self.selected_positive_count = 0
        self.candidate_count = 0
        self.disputed_count = 0
        self.sample_totals = {seed: [0, 0, 0] for seed in self.seeds}
        self.strata: dict[
            str, dict[int | None, dict[str, Any]]
        ] = {dimension: {} for dimension in self.stratify}

    def add(self, record: Mapping[str, Any]) -> None:
        self.query_count += 1
        if record["status"] == "audited":
            self.audited_query_count += 1
        else:
            self.no_reachable_answer_count += 1
        self.selected_positive_count += int(record["selected_positive_count"])
        self.candidate_count += int(record["candidate_pool_size"])
        self.disputed_count += int(record["disputed_candidate_count"])
        for sample in record["sampling"]:
            totals = self.sample_totals[int(sample["seed"])]
            totals[0] += int(sample["requested_sample_count"])
            totals[1] += int(sample["sampled_count"])
            totals[2] += int(sample["disputed_sampled_count"])

        for dimension in self.stratify:
            seen_values: set[int | None] = set()
            for row in record["strata"].get(dimension, []):
                value = row["value"]
                aggregate = self.strata[dimension].setdefault(
                    value,
                    {
                        "query_count": 0,
                        "candidate_pool_size": 0,
                        "disputed_candidate_count": 0,
                        "samples": {seed: [0, 0] for seed in self.seeds},
                    },
                )
                if value not in seen_values:
                    aggregate["query_count"] += 1
                    seen_values.add(value)
                aggregate["candidate_pool_size"] += int(row["candidate_pool_size"])
                aggregate["disputed_candidate_count"] += int(
                    row["disputed_candidate_count"]
                )
                for sample in row["sampling_by_seed"]:
                    totals = aggregate["samples"][int(sample["seed"])]
                    totals[0] += int(sample["sampled_count"])
                    totals[1] += int(sample["disputed_sampled_count"])

    def _strata_report(self) -> dict[str, list[dict[str, Any]]]:
        report: dict[str, list[dict[str, Any]]] = {}
        for dimension, values in self.strata.items():
            rows: list[dict[str, Any]] = []
            for value in sorted(values, key=lambda item: (-1 if item is None else item)):
                aggregate = values[value]
                candidate_count = aggregate["candidate_pool_size"]
                disputed_count = aggregate["disputed_candidate_count"]
                rows.append(
                    {
                        "value": value,
                        "query_count": aggregate["query_count"],
                        "candidate_pool_size": candidate_count,
                        "disputed_candidate_count": disputed_count,
                        "disputed_candidate_rate": _ratio(
                            disputed_count, candidate_count
                        ),
                        "sampling_by_seed": [
                            {
                                "seed": seed,
                                "sampled_count": aggregate["samples"][seed][0],
                                "disputed_sampled_count": aggregate["samples"][seed][1],
                                "disputed_sampled_rate": _ratio(
                                    aggregate["samples"][seed][1],
                                    aggregate["samples"][seed][0],
                                ),
                            }
                            for seed in self.seeds
                        ],
                    }
                )
            report[dimension] = rows
        return report

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "domain": self.domain.name,
            "query_shape": ["e", ["r", "r", "r"]],
            "seeds": list(self.seeds),
            "stratification_dimensions": list(self.stratify),
            "stratification_definitions": {
                "hop": "maximum logical-hop length among selected answer paths",
                "arity": "number of unique entities incident to the candidate hyperedge",
                "depth": "full-graph shortest logical-hop distance from topic to candidate head",
            },
            "graph": {
                "raw_triple_count": self.domain.triple_count,
                "unique_triple_count": self.domain.unique_triple_count,
                "entity_count": self.domain.entity_count,
                "hyperedge_count": self.domain.hyperedge_count,
                "incidence_edge_count": self.domain.graph.number_of_edges(),
                "aggregation": "one n-ary hyperedge per train_graph head",
            },
            "mapping_sizes": dict(iter_mapping_sizes(self.domain.mappings)),
            "source_three_hop_query_count": len(self.domain.examples),
            "query_count": self.query_count,
            "audited_query_count": self.audited_query_count,
            "no_reachable_answer_count": self.no_reachable_answer_count,
            "selected_positive_count": self.selected_positive_count,
            "candidate_pool_size": self.candidate_count,
            "disputed_candidate_count": self.disputed_count,
            "disputed_candidate_rate": _ratio(
                self.disputed_count, self.candidate_count
            ),
            "sampling_by_seed": [
                {
                    "seed": seed,
                    "requested_sample_count": self.sample_totals[seed][0],
                    "sampled_count": self.sample_totals[seed][1],
                    "disputed_sampled_count": self.sample_totals[seed][2],
                    "disputed_sampled_rate": _ratio(
                        self.sample_totals[seed][2], self.sample_totals[seed][1]
                    ),
                }
                for seed in self.seeds
            ],
            "strata": self._strata_report(),
        }


def audit_domain_to_files(
    domain_dir: Path,
    jsonl_path: Path,
    summary_path: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    stratify: Sequence[str] = STRATIFICATION_DIMENSIONS,
) -> dict[str, Any]:
    """Run a domain audit and atomically replace its two result files."""

    domain = load_domain(domain_dir)
    accumulator = SummaryAccumulator(domain, seeds, stratify)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_temp = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    summary_temp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    try:
        with jsonl_temp.open("w", encoding="utf-8", newline="\n") as handle:
            for query_index, example in enumerate(domain.examples):
                record = audit_example(
                    domain,
                    example,
                    query_index,
                    seeds=seeds,
                    stratify=stratify,
                )
                accumulator.add(record)
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        report = accumulator.report()
        with summary_temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        jsonl_temp.replace(jsonl_path)
        summary_temp.replace(summary_path)
    finally:
        jsonl_temp.unlink(missing_ok=True)
        summary_temp.unlink(missing_ok=True)
    return accumulator.report()


def _discover_domains(dataset_root: Path) -> tuple[str, ...]:
    return tuple(
        path.name
        for path in sorted(dataset_root.iterdir(), key=lambda value: value.name)
        if path.is_dir() and (path / "train_graph.txt").is_file()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit path-consistent candidate negatives directly from public "
            "WikiTopics_QE integer-ID files."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory containing WikiTopics_QE/<domain>",
    )
    parser.add_argument(
        "--domain",
        action="append",
        help="Domain to audit; repeat this option, or omit it to discover all domains",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Writes <domain>.audit.jsonl and <domain>.summary.json here",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Shared sampler seeds (default: 42 43 44 45 46)",
    )
    parser.add_argument(
        "--stratify",
        nargs="*",
        choices=STRATIFICATION_DIMENSIONS,
        default=list(STRATIFICATION_DIMENSIONS),
        help="Summary dimensions; pass with no values to disable stratification",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    domains = tuple(dict.fromkeys(args.domain or _discover_domains(args.dataset_root)))
    if not domains:
        raise SystemExit("No WikiTopics domains found")

    compact_reports = []
    for domain_name in domains:
        domain_dir = args.dataset_root / domain_name
        report = audit_domain_to_files(
            domain_dir=domain_dir,
            jsonl_path=args.output_dir / f"{domain_name}.audit.jsonl",
            summary_path=args.output_dir / f"{domain_name}.summary.json",
            seeds=args.seeds,
            stratify=args.stratify,
        )
        compact_reports.append(
            {
                "domain": domain_name,
                "query_count": report["query_count"],
                "candidate_pool_size": report["candidate_pool_size"],
                "disputed_candidate_count": report["disputed_candidate_count"],
                "disputed_candidate_rate": report["disputed_candidate_rate"],
            }
        )
    print(json.dumps(compact_reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
