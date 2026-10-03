"""Export observed PCN count distributions, retaining seed and domain units."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def affected_median(hist):
    total = sum(n for x, n in hist.items() if x > 0)
    if not total:
        return None
    targets = ((total - 1) // 2, total // 2)
    values, seen = [], 0
    for x, n in sorted(hist.items()):
        if x == 0:
            continue
        values.extend(x for target in targets if seen <= target < seen + n)
        seen += n
    return sum(values) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/cross_dataset_20261003")
    root = parser.parse_args().artifacts
    wiki = read(root / "wiki_prevalence_compact.json")
    groups = []
    for row in wiki:
        groups.append(("WikiTopics", row["dataset"], row["per_seed"], row["per_seed_histograms"]))
    # Wiki pooled statistics describe questions, not equal-domain macro values.
    pooled = {}
    pooled_rows = []
    for i, seed in enumerate(wiki[0]["seeds"]):
        hist = Counter()
        for domain in wiki:
            hist.update({int(k): v for k, v in domain["per_seed_histograms"][str(seed)].items()})
        pooled[str(seed)] = hist
        affected = sum(v for k, v in hist.items() if k > 0)
        count = sum(k * v for k, v in hist.items())
        pooled_rows.append({"seed": seed, "query_count": sum(hist.values()),
            "sampled_negatives": sum(d["per_seed"][i]["sampled_negatives"] for d in wiki),
            "pcn_count": count, "affected_questions": affected,
            "mean_pcn_per_affected": count / affected, "median_pcn_per_affected": affected_median(hist)})
    groups.append(("WikiTopics", "pooled_questions", pooled_rows, pooled))
    meta = read(root / "metaqa_prevalence.json")
    meta_hist = meta["per_seed_histograms"]
    groups.append(("MetaQA-3hop-vanilla", "all", meta["per_seed"], meta_hist))
    for name in ("pathquestion", "kqapro"):
        data = read(root / f"{name}_inspection.json")
        histograms, rows = {}, []
        for i, seed in enumerate((42, 43, 44, 45, 46)):
            hist = Counter(q["pcn_per_seed"][i] for q in data["queries"])
            histograms[str(seed)] = hist
            affected = sum(n for x, n in hist.items() if x > 0)
            pcn = sum(x * n for x, n in hist.items())
            rows.append({"seed": seed, "query_count": len(data["queries"]),
                "sampled_negatives": data["per_seed"][str(seed)]["sampled_negatives"],
                "pcn_count": pcn, "affected_questions": affected,
                "mean_pcn_per_affected": pcn / affected if affected else None,
                "median_pcn_per_affected": affected_median(hist)})
        groups.append((name, "all", rows, histograms))
    stats, distribution = [], []
    for dataset, domain, rows, histograms in groups:
        for row in rows:
            stats.append({"dataset": dataset, "domain": domain, **row})
            for x, n in sorted((int(x), n) for x, n in histograms[str(row["seed"])].items()):
                distribution.append({"dataset": dataset, "domain": domain, "seed": row["seed"],
                    "pcn_per_question": x, "question_count": n})
    csv_write(root / "prevalence_per_seed.csv", stats)
    csv_write(root / "prevalence_count_distribution.csv", distribution)
    print("Exported real per-seed PCN statistics and count histograms")


if __name__ == "__main__":
    main()
