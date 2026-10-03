"""Build reusable six-GPU manifests without rewriting existing runs."""

import argparse
from pathlib import Path
from research.path_consistent_negative_learning.scripts.run_experiment import write_json, SEEDS, STRATEGIES

DOMAINS = ("art", "award", "edu", "health", "infra", "loc", "org", "people", "sci", "sport", "tax")
RUNNER = "research.path_consistent_negative_learning.scripts.run_experiment"


def job(identifier, commands, output, expected):
    return {"job_id": identifier, "commands": commands, "output_dir": str(output),
            "expected_files": [str(x) for x in expected]}


def build(phase, base):
    root = base / "runs/cross_dataset_20261003"
    sources = base / "data_sources/pcn_cross_dataset_20261003"
    prefix = ["{python}", "-m", RUNNER]
    jobs = []
    if phase == "all-shortest":
        for dataset in ("pathquestion", "kqapro"):
            work = root / dataset
            output = work / "all_shortest"
            prepared = output / "prepared"
            commands = [prefix + ["prepare", "--dataset", dataset,
                        "--data-root", str(sources/dataset), "--embeddings", str(work/"embeddings.pt"),
                        "--output-dir", str(prepared), "--split", "train", "--all-shortest", "--device", "cuda:0"]]
            for seed in SEEDS:
                commands.append(prefix + ["train", "--prepared", str(prepared/f"train_seed_{seed}.pt"),
                                "--embeddings", str(work/"embeddings.pt"), "--evaluation", str(work/"prepared/test.pt"),
                                "--output-dir", str(output/f"seed_{seed}"), "--strategy", "baseline", "--seed", str(seed),
                                "--batch-size", "512", "--feature-storage", "indexed", "--device", "cuda:0",
                                "--source-protocol", "all_shortest_positive_union_new_negative_pool"])
            item = job(f"{dataset}_all_shortest", commands, output,
                       [output/f"seed_{s}/metrics.json" for s in SEEDS])
            item["maximum_used_mib"] = 18000
            jobs.append(item)
        return {"phase": phase, "jobs": jobs}
    if phase == "main":
        preparation = build("prepare-new", base)["jobs"]
        training = build("train-new", base)["jobs"]
        for item in training:
            dataset = item["job_id"].split("_", 1)[0]
            item["depends_on"] = [f"{dataset}_prepare_train", f"{dataset}_prepare_test"]
        return {"phase": phase, "jobs": preparation + training + build("existing-positive", base)["jobs"]}
    if phase == "existing-positive":
        # Launch large WikiTopics jobs first to avoid a long final GPU tail.
        for domain in ("people", "loc", "org", "art", "sci", "infra", "award", "edu", "health", "sport", "tax"):
            for seed in SEEDS:
                output = root / "WikiTopics" / domain / "positive_relabel" / f"seed_{seed}"
                old = base / "runs/final_revision_answer_free"
                commands = [prefix + ["train", "--prepared", str(old / f"prepared/train/{domain}/seed_{seed}.pt"),
                           "--embeddings", str(base / f"runs/retriever_only/embeddings/{domain}.pt"),
                           "--evaluation", str(old / f"prepared/eval/{domain}/test.pt"),
                           "--output-dir", str(output), "--strategy", "positive_relabel", "--seed", str(seed),
                           "--batch-size", "32", "--feature-storage", "indexed", "--device", "cuda:0",
                           "--source-protocol", "WikiTopics_fixed_original_config"]]
                jobs.append(job(f"wiki_{domain}_relabel_{seed}", commands, output, [output/"metrics.json"]))
        for seed in SEEDS:
            output = root / "MetaQA-3hop-vanilla/positive_relabel" / f"seed_{seed}"
            old = base / "runs/final_confirmation_metaqa"
            commands = [prefix + ["train", "--prepared", str(old / f"prepared/train_seed_{seed}.pt"),
                       "--embeddings", str(old / "embeddings_train_dev.pt"),
                       "--evaluation-embeddings", str(old / "embeddings_test.pt"),
                       "--evaluation", str(old / "prepared/test.pt"), "--output-dir", str(output),
                       "--strategy", "positive_relabel", "--seed", str(seed), "--batch-size", "512",
                       "--feature-storage", "indexed", "--device", "cuda:0", "--source-protocol", "MetaQA_fixed_original_config"]]
            jobs.append(job(f"metaqa_relabel_{seed}", commands, output, [output/"metrics.json"]))
    else:
        for dataset in ("pathquestion", "kqapro"):
            work = root / dataset
            if phase == "encode-new":
                for shard in range(3):
                    output = work / "encoding" / str(shard)
                    path = output / "embeddings.pt"
                    commands = [prefix + ["encode", "--dataset", dataset, "--data-root", str(sources/dataset),
                               "--model", str(base/"data_sources/models/gte-large-en-v1.5"),
                               "--output", str(path), "--shards", "3", "--shard-id", str(shard),
                               "--encoding-batch-size", "64", "--device", "cuda:0"]]
                    jobs.append(job(f"{dataset}_encode_{shard}", commands, output, [path]))
            elif phase == "prepare-new":
                for split in ("train", "valid", "test"):
                    output = work / "prepare_logs" / split
                    path = work / "prepared" / ("train_seed_46.pt" if split=="train" else f"{split}.pt")
                    commands = [prefix + ["prepare", "--dataset", dataset, "--data-root", str(sources/dataset),
                               "--embeddings", str(work/"embeddings.pt"), "--output-dir", str(work/"prepared"),
                               "--split", split, "--device", "cuda:0"]]
                    jobs.append(job(f"{dataset}_prepare_{split}", commands, output, [path]))
            elif phase == "train-new":
                for strategy in STRATEGIES:
                    for seed in SEEDS:
                        output = work / strategy / f"seed_{seed}"
                        commands = [prefix + ["train", "--prepared", str(work/f"prepared/train_seed_{seed}.pt"),
                                   "--embeddings", str(work/"embeddings.pt"), "--evaluation", str(work/"prepared/test.pt"),
                                   "--output-dir", str(output), "--strategy", strategy, "--seed", str(seed),
                                   "--batch-size", "512", "--feature-storage", "indexed" if dataset=="kqapro" else "materialized",
                                   "--device", "cuda:0"]]
                        jobs.append(job(f"{dataset}_{strategy}_{seed}", commands, output, [output/"metrics.json"]))
    return {"phase": phase, "jobs": jobs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("existing-positive", "encode-new", "prepare-new", "train-new", "main", "all-shortest"), required=True)
    parser.add_argument("--base", type=Path, default=Path("/root/hyperrag_pcneg"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.phase, args.base)
    write_json(args.output, manifest)
    print(f"{args.phase}: {len(manifest['jobs'])} jobs")


if __name__ == "__main__":
    main()
