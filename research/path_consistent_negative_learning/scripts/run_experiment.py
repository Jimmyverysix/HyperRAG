"""Shared PCN dataset preparation, four-strategy training and evaluation CLI."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

from research.path_consistent_negative_learning.datasets.ordinary import (
    load_dataset, training_structure, sample_training_transitions, shortest_tree,
)
from research.path_consistent_negative_learning.retriever_only.embeddings import (
    EmbeddingStore, OfficialGTEEncoder,
)
from research.path_consistent_negative_learning.retriever_only.official import OfficialDDE
from research.path_consistent_negative_learning.retriever_only.preparation import _Accumulator
from research.path_consistent_negative_learning.retriever_only.prepared import PreparedCandidates
from research.path_consistent_negative_learning.retriever_only.training import TrainingConfig, train_retriever
from research.path_consistent_negative_learning.retriever_only.evaluation import score_candidates, evaluate_scores
from research.path_consistent_negative_learning.retriever_only.metrics import answer_path_metrics

SEEDS = (42, 43, 44, 45, 46)
STRATEGIES = {"baseline": "baseline", "random_mask": "matched_random",
              "pcn_mask": "ours", "positive_relabel": "positive_relabel"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def encode(args):
    adapter = load_dataset(args.dataset, args.data_root)
    nodes = tuple(sorted(adapter.graph.node_texts))
    questions = tuple(sorted({q.text for qs in adapter.splits.values() for q in qs}))
    node_slice = nodes[args.shard_id::args.shards]
    question_slice = questions[args.shard_id::args.shards]
    encoder = OfficialGTEEncoder(args.model, args.device)
    EmbeddingStore(node_slice, encoder.encode([adapter.graph.node_texts[n] for n in node_slice],
                                             batch_size=args.encoding_batch_size),
                   question_slice, encoder.encode(question_slice, batch_size=args.encoding_batch_size)
                   ).save(args.output)


def merge_embeddings(args):
    stores = [EmbeddingStore.load(p) for p in args.inputs]
    node_names = [name for s in stores for name in s.node_names]
    question_texts = [text for s in stores for text in s.query_texts]
    node_order = sorted(range(len(node_names)), key=node_names.__getitem__)
    question_order = sorted(range(len(question_texts)), key=question_texts.__getitem__)
    EmbeddingStore(tuple(node_names[i] for i in node_order),
                   torch.cat([s.node_embeddings for s in stores])[node_order],
                   tuple(question_texts[i] for i in question_order),
                   torch.cat([s.query_embeddings for s in stores])[question_order]).save(args.output)


def query_diagnostics(adapter, query, structure):
    positives, _, distances, on_path = structure
    counts = {query.topic: 1}
    for head in sorted(distances, key=lambda n: (distances[n], n)):
        for _, tail in adapter.graph.outgoing.get(head, ()):
            if distances.get(tail) == distances[head] + 1:
                counts[tail] = counts.get(tail, 0) + counts.get(head, 0)
    answer_counts = [counts[a] for a in query.answers if a in counts and distances[a] > 0]
    hops = [distances[a] for a in query.answers if a in distances and distances[a] > 0]
    return {"query_key": query.key, "maximum_answer_hops": max(hops, default=0),
            "shortest_path_count": sum(answer_counts),
            "alternative_shortest_path_count": sum(max(0, n - 1) for n in answer_counts),
            "multiple_shortest": any(n > 1 for n in answer_counts),
            "relation_diversity": len({r for _, r, _ in positives}),
            "selected_positive_count": len(positives)}


def prepare(args):
    adapter = load_dataset(args.dataset, args.data_root)
    embeddings = EmbeddingStore.load(args.embeddings)
    encoder = OfficialDDE(device=args.device, maximum_hops=3)
    output = args.output_dir
    graph_stats = {**adapter.metadata, "dataset": adapter.name,
                   "entity_count": sum(n.startswith("E:") for n in adapter.graph.node_texts),
                   "relation_text_count_with_inverse": sum(n.startswith("R:") for n in adapter.graph.node_texts),
                   "triple_count": adapter.graph.triple_count,
                   "split_sizes": {k: len(v) for k, v in adapter.splits.items()}}
    write_json(output / "dataset.json", graph_stats)
    if args.split == "train":
        accumulators = {s: _Accumulator(domain=adapter.name, split="train", seed=s) for s in SEEDS}
        diagnostics, cache, counts = [], {}, []
        for i, query in enumerate(adapter.load_train()):
            structure = training_structure(adapter.graph, query, topic_cache=cache,
                                           all_shortest=args.all_shortest)
            if structure is None:
                continue
            diagnostics.append(query_diagnostics(adapter, query, structure))
            seed_counts = []
            for seed, accumulator in accumulators.items():
                positives, negatives, conflicts = sample_training_transitions(
                    structure, seed=seed, query_key=query.key)
                candidates = positives + negatives
                accumulator.add(key=query.key, text=query.text, topic=query.topic, answers=query.answers,
                                candidates=candidates, dde=encoder.encode(candidates, query.topic),
                                embeddings=embeddings, labels=[True]*len(positives)+[False]*len(negatives),
                                path_mask=[False]*len(positives)+list(conflicts))
                seed_counts.append({"seed": seed, "negative_count": len(negatives),
                                    "pcn_count": sum(conflicts)})
            counts.append({"query_key": query.key, "seeds": seed_counts})
            if (i + 1) % 1000 == 0:
                print(f"{adapter.name} train {i+1}/{len(adapter.load_train())}", flush=True)
        for seed, accumulator in accumulators.items():
            accumulator.finish().save(output / f"train_seed_{seed}.pt")
        write_json(output / "train_diagnostics.json", {"dataset": adapter.name,
                    "official_train_subset_count": len(adapter.load_train()),
                    "eligible_train_count": len(counts), "queries": diagnostics})
        write_json(output / "sampled_negative_counts.json", {"queries": counts})
    else:
        accumulator = _Accumulator(domain=adapter.name, split=args.split, seed=None,
                                   labels=None, path_consistent_mask=None, transitions=[])
        nodes = embeddings.node_embeddings.to(args.device)
        node_index, query_index = embeddings.node_index, embeddings.query_index
        for i, query in enumerate(adapter.splits[args.split]):
            # All selected evaluation questions remain, including graph-outside
            # answers and empty candidate pools. Answers cannot gate retrieval.
            vector = embeddings.query_embeddings[query_index[query.text]].to(args.device)
            with torch.no_grad():
                similarities = torch.mv(nodes, vector).cpu().tolist()
            candidates = adapter.get_candidate_transitions(
                query.topic, lambda n: similarities[node_index[n]], width=32)
            accumulator.add(key=query.key, text=query.text, topic=query.topic, answers=query.answers,
                            candidates=candidates, dde=encoder.encode(candidates, query.topic),
                            embeddings=embeddings, labels=None, path_mask=None)
            if (i + 1) % 1000 == 0:
                print(f"{adapter.name} {args.split} {i+1}/{len(adapter.splits[args.split])}", flush=True)
        accumulator.finish().save(output / f"{args.split}.pt")


def train(args):
    if (args.output_dir / "metrics.json").exists() or (args.output_dir / "checkpoint.pt").exists():
        raise FileExistsError(f"existing run is preserved: {args.output_dir}")
    started = datetime.now(timezone.utc).isoformat()
    data = PreparedCandidates.load(args.prepared)
    if data.seed != args.seed:
        raise ValueError("prepared sampling seed must equal training seed")
    embeddings = EmbeddingStore.load(args.embeddings)
    config = TrainingConfig(batch_size=args.batch_size, maximum_epochs=args.maximum_epochs,
                            feature_storage=args.feature_storage)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(args.output_dir / "config.json", {"dataset": data.domain, "strategy": args.strategy,
               "seed": args.seed, "training": asdict(config), "git_commit": commit,
               "command": sys.argv, "python": sys.version, "torch": torch.__version__,
               "platform": platform.platform(), "device": args.device,
               "prepared": str(args.prepared), "embeddings": str(args.embeddings),
               "started_utc": started, "source_protocol": args.source_protocol})
    report = train_retriever(data, embeddings, method=STRATEGIES[args.strategy], lambda_=0.0,
                            seed=args.seed, device_name=args.device,
                            checkpoint_path=args.output_dir / "checkpoint.pt", config=config)
    report.update(started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output_dir / "training.json", report)
    if args.evaluation is not None:
        evaluate(args)


def evaluate(args):
    data = PreparedCandidates.load(args.evaluation)
    embeddings = EmbeddingStore.load(args.evaluation_embeddings or args.embeddings)
    checkpoint = args.checkpoint or args.output_dir / "checkpoint.pt"
    scores = score_candidates(data, embeddings, checkpoint, device_name=args.device, batch_size=4096)
    report = evaluate_scores(data, scores)
    for i, row in enumerate(report["queries"]):
        start, stop = int(data.query_offsets[i]), int(data.query_offsets[i+1])
        row["candidate_oracle"] = int(answer_path_metrics(data.transitions[start:stop],
                                       data.query_topics[i], data.query_answers[i]).first_answer_rank is not None)
    report["strategy"], report["seed"] = args.strategy, args.seed
    report["evaluated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output_dir / "metrics.json", report)


def prevalence(args):
    per_seed, counts_by_key = [], {}
    negatives = conflicts = 0
    for seed in SEEDS:
        path = args.prepared_dir / args.filename_pattern.format(seed=seed)
        data = PreparedCandidates.load(path)
        n = int((~data.labels).sum())
        d = int(data.path_consistent_mask.sum())
        qcounts = []
        for i, key in enumerate(data.query_keys):
            a, b = int(data.query_offsets[i]), int(data.query_offsets[i+1])
            count = int(data.path_consistent_mask[a:b].sum())
            qcounts.append(count)
            counts_by_key.setdefault(key, []).append(count)
        affected = [n for n in qcounts if n]
        negatives += n
        conflicts += d
        per_seed.append({"seed": seed, "query_count": len(qcounts), "sampled_negatives": n,
                         "pcn_count": d, "affected_questions": len(affected),
                         "mean_pcn_per_affected": float(np.mean(affected)) if affected else 0,
                         "median_pcn_per_affected": float(np.median(affected)) if affected else 0})
    affected_any = sum(any(v) for v in counts_by_key.values())
    write_json(args.output, {"dataset": args.dataset, "train_questions": len(counts_by_key),
               "seeds": list(SEEDS), "sampled_negatives_seed_sum": negatives,
               "pcn_seed_sum": conflicts, "pcn_ratio": conflicts/negatives if negatives else 0,
               "affected_questions_any_seed": affected_any,
               "affected_ratio_any_seed": affected_any/len(counts_by_key),
               "per_seed": per_seed, "per_question_pcn": counts_by_key})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("encode", "prepare"):
        p = sub.add_parser(command)
        p.add_argument("--dataset", choices=("metaqa", "kqapro", "pathquestion"), required=True)
        p.add_argument("--data-root", type=Path, required=True)
        p.add_argument("--device", default="cuda:0")
        if command == "encode":
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--shards", type=int, default=1)
            p.add_argument("--shard-id", type=int, default=0)
            p.add_argument("--encoding-batch-size", type=int, default=64)
        else:
            p.add_argument("--embeddings", type=Path, required=True)
            p.add_argument("--split", choices=("train", "valid", "test"), required=True)
            p.add_argument("--output-dir", type=Path, required=True)
            p.add_argument("--all-shortest", action="store_true")
    p = sub.add_parser("merge-embeddings")
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    for command in ("train", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--prepared", type=Path, required=command == "train")
        p.add_argument("--embeddings", type=Path, required=True)
        p.add_argument("--evaluation", type=Path, required=command == "evaluate")
        p.add_argument("--evaluation-embeddings", type=Path)
        p.add_argument("--checkpoint", type=Path)
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--strategy", choices=tuple(STRATEGIES), required=True)
        p.add_argument("--seed", type=int, choices=SEEDS, required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--batch-size", type=int, default=512)
        p.add_argument("--maximum-epochs", type=int, default=50)
        p.add_argument("--feature-storage", choices=("indexed", "materialized"), default="indexed")
        p.add_argument("--source-protocol", default="pcn_cross_dataset_20261003")
    p = sub.add_parser("prevalence")
    p.add_argument("--dataset", required=True)
    p.add_argument("--prepared-dir", type=Path, required=True)
    p.add_argument("--filename-pattern", default="train_seed_{seed}.pt")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Many simultaneous workers use independent GPUs; avoid 64-thread CPU
    # oversubscription in each tiny MLP or per-question preprocessing worker.
    torch.set_num_threads(1)
    {"encode": encode, "merge-embeddings": merge_embeddings, "prepare": prepare,
     "train": train, "evaluate": evaluate, "prevalence": prevalence}[args.command](args)


if __name__ == "__main__":
    main()
