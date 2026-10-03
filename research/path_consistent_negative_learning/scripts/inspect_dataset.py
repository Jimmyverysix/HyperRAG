"""Train-only compatibility and actual-sampler prevalence for new datasets."""

import argparse
from collections import Counter
from pathlib import Path
import numpy as np

from research.path_consistent_negative_learning.datasets.ordinary import load_dataset, training_structure, sample_training_transitions
from research.path_consistent_negative_learning.scripts.run_experiment import SEEDS, query_diagnostics, write_json


def inspect(adapter):
    cache, rows, distributions = {}, [], Counter()
    affected = set()
    negatives = pcn = 0
    seed_totals = {s: {"sampled_negatives": 0, "pcn": 0, "affected": 0} for s in SEEDS}
    for index, q in enumerate(adapter.load_train()):
        structure = training_structure(adapter.graph, q, topic_cache=cache)
        if structure is None:
            continue
        diagnosis = query_diagnostics(adapter, q, structure)
        seed_counts = []
        for seed in SEEDS:
            _, sampled, mask = sample_training_transitions(structure, seed=seed, query_key=q.key)
            count = sum(mask)
            negatives += len(sampled)
            pcn += count
            seed_totals[seed]["sampled_negatives"] += len(sampled)
            seed_totals[seed]["pcn"] += count
            seed_totals[seed]["affected"] += int(count > 0)
            seed_counts.append(count)
        if any(seed_counts):
            affected.add(q.key)
        diagnosis["pcn_per_seed"] = seed_counts
        diagnosis["pcn_mean"] = float(np.mean(seed_counts))
        diagnosis["negative_count"] = len(sampled)
        rows.append(diagnosis)
        distributions[diagnosis["maximum_answer_hops"]] += 1
        if (index + 1) % 5000 == 0:
            print(adapter.name, index+1, flush=True)
    return {"dataset": adapter.name, "metadata": adapter.metadata,
            "split_sizes": {k: len(v) for k, v in adapter.splits.items()},
            "graph_entity_count": sum(n.startswith("E:") for n in adapter.graph.node_texts),
            "graph_triples": adapter.graph.triple_count,
            "directed_edges_with_inverse": sum(len(v) for v in adapter.graph.outgoing.values()),
            "relation_count_with_inverse": sum(n.startswith("R:") for n in adapter.graph.node_texts),
            "eligible_train_questions": len(rows),
            "sampled_negatives_seed_sum": negatives, "pcn_seed_sum": pcn,
            "pcn_ratio": pcn / negatives if negatives else 0,
            "affected_questions_any_seed": len(affected),
            "affected_ratio_any_seed": len(affected)/len(rows) if rows else 0,
            "shortest_hop_distribution_train": dict(sorted(distributions.items())),
            "multiple_shortest_questions_train": sum(r["multiple_shortest"] for r in rows),
            "per_seed": seed_totals, "queries": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=("kqapro", "pathquestion", "metaqa"))
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = inspect(load_dataset(a.dataset, a.data_root))
    write_json(a.output, report)
    print({k: v for k, v in report.items() if k not in ("queries", "metadata")})


if __name__ == "__main__":
    main()
