"""Post-evaluation structure diagnostics; gold answers never alter retrieval."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .aggregate_cross_dataset_results import average_reports, SEEDS
from .run_experiment import query_diagnostics, write_json
from ..datasets.ordinary import load_dataset, training_structure, shortest_tree
from ..retriever_only.embeddings import EmbeddingStore
from ..retriever_only.prepared import PreparedCandidates
from ..retriever_only.evaluation import score_candidates


def summarize_rows(rows, field, boundaries):
    result = []
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        selected = [r for r in rows if r.get(field) is not None and low <= r[field] < high]
        if selected:
            result.append({"lower": low, "upper": high if np.isfinite(high) else None,
                           "question_count": len(selected),
                           "mean_apc_delta": float(np.mean([r["apc_delta"] for r in selected])),
                           "mean_reach10_delta": float(np.mean([r["reach10_delta"] for r in selected]))})
    return result


def analyze(dataset, base, output):
    work = base / "runs/cross_dataset_20261003" / dataset
    adapter = load_dataset(dataset, base/"data_sources/pcn_cross_dataset_20261003"/dataset)
    ours = average_reports([work/f"pcn_mask/seed_{s}/metrics.json" for s in SEEDS])
    baseline = average_reports([work/f"baseline/seed_{s}/metrics.json" for s in SEEDS])
    if ours is None or baseline is None or ours["keys"] != baseline["keys"]:
        raise ValueError("five paired seeds are required before mechanism analysis")
    deltas = {k: values for k, values in zip(ours["keys"], ours["values"] - baseline["values"])}
    training_counts = __import__('json').loads((work/"prepared/sampled_negative_counts.json").read_text())["queries"]
    count_by_key = {r["query_key"]: r["seeds"] for r in training_counts}
    topic_counts = defaultdict(lambda: [0, 0, 0])
    for q in adapter.load_train():
        if q.key in count_by_key:
            counts = count_by_key[q.key]
            topic_counts[q.topic][0] += sum(r["pcn_count"] for r in counts)
            topic_counts[q.topic][1] += sum(r["negative_count"] for r in counts)
            topic_counts[q.topic][2] += len(counts)
    rows, cache = [], {}
    for query in adapter.load_test():
        structure = training_structure(adapter.graph, query, topic_cache=cache)
        diagnostic = query_diagnostics(adapter, query, structure) if structure else {
            "maximum_answer_hops": 0, "shortest_path_count": 0,
            "alternative_shortest_path_count": 0, "relation_diversity": 0}
        counts = topic_counts.get(query.topic)
        rows.append({**diagnostic, "query_key": query.key,
            "apc_delta": float(deltas[query.key][0]), "reach10_delta": float(deltas[query.key][1]),
            "candidate_count": ours["candidate_counts"][query.key],
            "oracle_reachable": ours["oracle"][query.key],
            "training_topic_pcn_ratio": counts[0]/counts[1] if counts and counts[1] else None,
            "training_topic_mean_pcn": counts[0]/counts[2] if counts else None})
    buckets = {
        "maximum_answer_hops": [0, 1, 2, 3, 4],
        "shortest_path_count": [0, 1, 2, 4, 16, float('inf')],
        "alternative_shortest_path_count": [0, 1, 2, 4, 16, float('inf')],
        "relation_diversity": [0, 1, 2, 4, 8, float('inf')],
        "candidate_count": [0, 16, 32, 64, 96, float('inf')],
        "oracle_reachable": [0, 1, 2],
        "training_topic_pcn_ratio": [0, .01, .05, .1, .25, 1.01],
        "training_topic_mean_pcn": [0, .1, .5, 1, 4, float('inf')],
    }
    # Rank ACTUAL training PCNs in the shared training pools. This measures
    # treatment behaviour, not held-out retrieval quality or causal mediation.
    embeddings = EmbeddingStore.load(work/"embeddings.pt")
    ranks = {m: [] for m in ("baseline", "pcn_mask")}
    ranks_seed = []
    for seed in SEEDS:
        data = PreparedCandidates.load(work/f"prepared/train_seed_{seed}.pt")
        row = {"seed": seed}
        for method in ranks:
            scores = score_candidates(data, embeddings, work/f"{method}/seed_{seed}/checkpoint.pt",
                                      device_name="cpu", batch_size=4096)
            seed_ranks = []
            for i, key in enumerate(data.query_keys):
                a, b = int(data.query_offsets[i]), int(data.query_offsets[i+1])
                mask = data.path_consistent_mask[a:b].numpy()
                if mask.any():
                    order = np.argsort(-scores[a:b].numpy(), kind="stable")
                    inverse = np.empty(len(order), dtype=int);inverse[order] = np.arange(1, len(order)+1)
                    seed_ranks.extend(inverse[mask].tolist())
            ranks[method].extend(seed_ranks)
            row[method] = {"pcn_count": len(seed_ranks), "mean_rank": float(np.mean(seed_ranks)),
                           "median_rank": float(np.median(seed_ranks)),
                           "top10_fraction": float(np.mean(np.asarray(seed_ranks) <= 10))}
        ranks_seed.append(row)
    output_data = {"dataset": adapter.name, "question_count": len(rows), "seeds": list(SEEDS),
        "interpretation": "descriptive associations; no causal mediation or per-bin significance claim",
        "test_answer_use": "post-ranking path diagnostics only; no candidate/feature/training changes",
        "training_topic_pcn_definition": "actual five-seed training negatives aggregated by topic; missing topics excluded from these two bins",
        "training_topic_matched_test_questions": sum(r['training_topic_mean_pcn'] is not None for r in rows),
        "bins": {f: summarize_rows(rows, f, b) for f, b in buckets.items()},
        "actual_training_pcn_rank": {m: {"count": len(v), "mean_rank": float(np.mean(v)),
            "median_rank": float(np.median(v)), "top10_fraction": float(np.mean(np.asarray(v) <= 10))}
            for m, v in ranks.items()}, "training_pcn_rank_per_seed": ranks_seed, "queries": rows}
    write_json(output, output_data)
    print(adapter.name, 'mechanism diagnostics complete', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=Path('/root/hyperrag_pcneg'))
    p.add_argument('--dataset', choices=('pathquestion','kqapro'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args();torch.set_num_threads(1);analyze(args.dataset, args.base, args.output)


if __name__ == '__main__':
    main()
