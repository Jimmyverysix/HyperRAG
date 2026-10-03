"""Aggregate paired five-seed PCN experiments; incomplete arms remain TODO."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEEDS = (42, 43, 44, 45, 46)
STRATEGIES = ("baseline", "random_mask", "pcn_mask", "positive_relabel")
METRICS = ("reciprocal_rank", "answer_reach_10", "answer_reach_5")
DOMAINS = ("art", "award", "edu", "health", "infra", "loc", "org", "people", "sci", "sport", "tax")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric_paths(root, relative):
    return [root/relative/f"seed_{seed}/metrics.json" if (root/relative/f"seed_{seed}/metrics.json").is_file()
            else root/"recovery_indexed"/relative/f"seed_{seed}/metrics.json" for seed in SEEDS]


def bootstrap(groups, resamples=10000, seed=20261003):
    """Exact empirical paired bootstrap, compressed by repeated difference rows.

    Multinomial counts sample the same empirical distribution as drawing n
    question indices with replacement. Metrics share each bootstrap draw.
    Each WikiTopics domain is resampled separately and equally weighted.
    """
    rng = np.random.default_rng(seed)
    distribution = np.zeros((resamples, next(iter(groups.values())).shape[1]))
    for values in groups.values():
        unique, counts = np.unique(values, axis=0, return_counts=True)
        for start in range(0, resamples, 100):
            stop = min(start + 100, resamples)
            draws = rng.multinomial(len(values), counts / len(values), size=stop-start)
            distribution[start:stop] += draws @ unique / len(values)
    distribution /= len(groups)
    mean = np.mean([v.mean(axis=0) for v in groups.values()], axis=0)
    low, high = np.quantile(distribution, [0.025, 0.975], axis=0)
    return {m: {"difference": float(mean[i]), "ci_low": float(low[i]),
                "ci_high": float(high[i])} for i, m in enumerate(METRICS)
            if np.isfinite(mean[i])}


def average_reports(paths, oracle=None):
    reports = [read_json(p) for p in paths if p.is_file()]
    by_seed = {int(r["seed"]): r for r in reports}
    if len(reports) != 5 or set(by_seed) != set(SEEDS):
        return None
    keys = sorted(q["query_key"] for q in reports[0]["queries"])
    arrays = []
    candidates = {}
    for seed in SEEDS:
        rows = {q["query_key"]: q for q in by_seed[seed]["queries"]}
        if sorted(rows) != keys:
            raise ValueError(f"query seed coverage differs: {paths[0]}")
        arrays.append([[float(rows[k].get(m, np.nan)) for m in METRICS] for k in keys])
        for key, row in rows.items():
            candidates[key] = int(row["candidate_count"])
            if "candidate_oracle" in row:
                if oracle is None:
                    oracle = {}
                current = int(row["candidate_oracle"])
                if key in oracle and oracle[key] != current:
                    raise ValueError(f"candidate oracle changed across seeds: {key}")
                oracle[key] = current
    return {"keys": keys, "values": np.asarray(arrays).mean(axis=0),
            "per_seed": {s: np.asarray(a).mean(axis=0) for s, a in zip(SEEDS, arrays)},
            "oracle": oracle, "candidate_counts": candidates,
            "sources": [str(p) for p in paths]}


def meta_reports(directory, old_method):
    rows = defaultdict(list)
    oracle, candidates = {}, {}
    paths = [directory / f"test_{old_method}_{seed}.csv" for seed in SEEDS]
    if not all(p.is_file() for p in paths):
        return None
    for seed, path in zip(SEEDS, paths):
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = row["query_id"]
                rr = float(row["APC_MRR_or_RR"])
                # RR is 1 / the integer completion rank, so Reach@5 is exact.
                rows[key].append([rr, float(row["Reach@10"]), float(rr >= 1/5)])
                oracle[key] = int(row["Candidate_Oracle"])
                candidates[key] = int(row["candidate_count"])
    if any(len(v) != 5 for v in rows.values()):
        raise ValueError("MetaQA query seed coverage differs")
    keys = sorted(rows)
    array = np.asarray([rows[k] for k in keys]).transpose(1, 0, 2)
    for seed, values in zip(SEEDS, array):
        recorded = read_json(directory/f"test_{old_method}_{seed}.json")["metrics"]["answer_reach_5"]
        if abs(float(values[:, 2].mean()) - recorded) > 1e-12:
            raise ValueError("Reach@5 reconstructed from RR differs from the recorded MetaQA report")
    return {"keys": keys, "values": array.mean(axis=0), "oracle": oracle,
            "candidate_counts": candidates,
            "per_seed": {s: a.mean(axis=0) for s, a in zip(SEEDS, array)},
            "sources": [str(p) for p in paths]}


def summarize(name, groups, resamples):
    output = {"dataset": name, "methods": {}, "comparisons": {}, "sources": {},
              "query_count": sum(len(next(v for v in arms.values() if v)["keys"]) for arms in groups.values()),
              "aggregation": "equal_domain_macro" if len(groups) > 1 else "question_mean"}
    complete = []
    for strategy in STRATEGIES:
        if any(arms.get(strategy) is None for arms in groups.values()):
            output["methods"][strategy] = {"status": "TODO", "reason": "five-seed arm incomplete"}
            continue
        complete.append(strategy)
        means, conditional, per_seed = [], [], {s: [] for s in SEEDS}
        for group, arms in groups.items():
            data = arms[strategy]
            means.append(data["values"].mean(axis=0))
            for seed in SEEDS:
                per_seed[seed].append(data["per_seed"][seed])
            reference = next((a for a in arms.values() if a and a["oracle"]), None)
            if reference:
                mask = np.asarray([bool(reference["oracle"][k]) for k in data["keys"]])
                if mask.any():
                    conditional.append(data["values"][mask].mean(axis=0))
            output["sources"].setdefault(strategy, {})[group] = data["sources"]
        mean = np.mean(means, axis=0)
        output["methods"][strategy] = {"status": "complete", "seeds": list(SEEDS),
            "all_questions": {m: float(v) for m, v in zip(METRICS, mean) if np.isfinite(v)},
            "per_seed": {str(s): {m: float(v) for m, v in zip(METRICS, np.mean(a, axis=0))
                                 if np.isfinite(v)} for s, a in per_seed.items()}}
        if len(conditional) == len(groups):
            output["methods"][strategy]["oracle_reachable_only"] = {
                m: float(v) for m, v in zip(METRICS, np.mean(conditional, axis=0)) if np.isfinite(v)}
    coverages, reachable_count = [], 0
    for group, arms in groups.items():
        reference = next((a for a in arms.values() if a and a["oracle"]), None)
        if reference:
            n = sum(reference["oracle"].values())
            coverages.append(n / len(reference["keys"]))
            reachable_count += n
    if len(coverages) == len(groups):
        output["candidate_oracle"] = float(np.mean(coverages))
        output["oracle_reachable_question_count"] = reachable_count
    if "pcn_mask" in complete:
        for strategy in complete:
            if strategy == "pcn_mask":
                continue
            differences, conditional = {}, {}
            for group, arms in groups.items():
                ours, control = arms["pcn_mask"], arms[strategy]
                if ours["keys"] != control["keys"]:
                    raise ValueError(f"paired methods have different questions: {name}/{group}")
                difference = ours["values"] - control["values"]
                differences[group] = np.nan_to_num(difference)
                reference = next((a for a in arms.values() if a and a["oracle"]), None)
                if reference:
                    mask = np.asarray([bool(reference["oracle"][k]) for k in ours["keys"]])
                    if mask.any():
                        conditional[group] = np.nan_to_num(difference[mask])
            result = bootstrap(differences, resamples=resamples)
            # Never turn an unavailable metric into a zero score.
            for m in METRICS:
                if m not in output["methods"][strategy]["all_questions"]:
                    result.pop(m, None)
            output["comparisons"][f"pcn_mask_minus_{strategy}"] = {"all_questions": result}
            if len(conditional) == len(groups):
                result = bootstrap(conditional, resamples=resamples)
                for m in METRICS:
                    if m not in output["methods"][strategy]["all_questions"]:
                        result.pop(m, None)
                output["comparisons"][f"pcn_mask_minus_{strategy}"]["oracle_reachable_only"] = result
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("/root/hyperrag_pcneg"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resamples", type=int, default=10000)
    args = p.parse_args()
    new = args.base / "runs/cross_dataset_20261003"
    old = args.base / "runs/final_revision_answer_free"
    selection = read_json(old / "selection/lambdas.json")
    groups = {}
    for domain in DOMAINS:
        fixed = args.base / f"runs/final_revision_answer_free/posthoc_fixed_masking/{domain}"
        # Preserve historic directory identifiers solely as input provenance.
        if not fixed.exists():
            fixed = args.base / f"runs/posthoc_fixed_masking/{domain}"
        same = float(selection["domains"][domain]["lambda"]) == 0.0
        directories = {"baseline": old/f"test/{domain}/baseline",
                       "random_mask": old/f"test/{domain}/matched_random" if same else fixed/"matched_random_masking",
                       "pcn_mask": old/f"test/{domain}/ours" if same else fixed/"fixed_masking"}
        groups[domain] = {s: average_reports([d/f"seed_{seed}/test_scores.report.json" for seed in SEEDS])
                          for s, d in directories.items()}
        d = new / f"WikiTopics/{domain}/positive_relabel"
        groups[domain]["positive_relabel"] = average_reports(metric_paths(new, f"WikiTopics/{domain}/positive_relabel"))
    summaries = [summarize("WikiTopics", groups, args.resamples)]
    domain_summaries = [summarize(d, {d: arms}, args.resamples) for d, arms in groups.items()]
    meta = {s: meta_reports(args.base/"runs/final_confirmation_metaqa/eval", old_method)
            for s, old_method in (("baseline", "baseline"), ("random_mask", "matched_random"), ("pcn_mask", "ours"))}
    meta["positive_relabel"] = average_reports(metric_paths(new, "MetaQA-3hop-vanilla/positive_relabel"))
    summaries.append(summarize("MetaQA-3hop-vanilla", {"metaqa": meta}, args.resamples))
    for dataset in ("pathquestion", "kqapro"):
        arms = {s: average_reports([new/f"{dataset}/{s}/seed_{seed}/metrics.json" for seed in SEEDS]) for s in STRATEGIES}
        summary = summarize(dataset, {dataset: arms}, args.resamples)
        optional = average_reports([new/f"{dataset}/all_shortest/seed_{s}/metrics.json" for s in SEEDS])
        if optional:
            ours = arms["pcn_mask"]
            if optional["keys"] != ours["keys"] or optional["candidate_counts"] != ours["candidate_counts"]:
                raise ValueError("all-shortest evaluation must share the original fixed candidate pool")
            mask = np.asarray([bool(optional["oracle"][k]) for k in optional["keys"]])
            summary["methods"]["all_shortest"] = {"status": "complete", "seeds": list(SEEDS),
                "training_candidates": "different_positive_union_and_new_negative_pool",
                "all_questions": dict(zip(METRICS, map(float, optional["values"].mean(axis=0)))),
                "oracle_reachable_only": dict(zip(METRICS, map(float, optional["values"][mask].mean(axis=0))))}
            summary["sources"]["all_shortest"] = {dataset: optional["sources"]}
            diff = ours["values"] - optional["values"]
            summary["comparisons"]["pcn_mask_minus_all_shortest"] = {
                "all_questions": bootstrap({dataset: diff}, args.resamples),
                "oracle_reachable_only": bootstrap({dataset: diff[mask]}, args.resamples)}
        summaries.append(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "seeds": list(SEEDS),
              "bootstrap": {"resamples": args.resamples, "confidence": .95, "seed": 20261003,
                            "unit": "question after averaging the five paired seeds",
                            "wiki": "resample within domain, then equal-domain macro"},
              "datasets": summaries, "wiki_domains": domain_summaries}
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    rows = [{"dataset": d["dataset"], "strategy": s, "scope": scope, "metric": m, "value": v}
            for d in summaries for s, a in d["methods"].items() if a["status"] == "complete"
            for scope in ("all_questions", "oracle_reachable_only") for m, v in a.get(scope, {}).items()]
    with (args.output_dir / "main_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("dataset", "strategy", "scope", "metric", "value"));w.writeheader();w.writerows(rows)
    for d in summaries:
        print(d["dataset"], {s: a["status"] for s, a in d["methods"].items()}, flush=True)


if __name__ == "__main__":
    main()
