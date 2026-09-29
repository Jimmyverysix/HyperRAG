"""Build reproducible GPU manifests for Retriever-only preparation and sweep."""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
from typing import Any


PIPELINE_MODULE = (
    "research.path_consistent_negative_learning.scripts.retriever_only_pipeline"
)
PREPARE_BATCH_MODULE = (
    "research.path_consistent_negative_learning.scripts."
    "prepare_retriever_training_batch"
)


def _lambda_token(value: float) -> str:
    return format(Decimal(str(value)), ".2f").replace(".", "p")


def _common_data(
    structured_root: Path,
    nlg_root: Path,
    label_snapshot: Path,
    domain: str,
) -> list[str]:
    return [
        "--structured-root",
        str(structured_root),
        "--nlg-root",
        str(nlg_root),
        "--label-snapshot",
        str(label_snapshot),
        "--domain",
        domain,
        "--device",
        "cuda",
    ]


def build_jobs(
    phase: str,
    config: dict[str, Any],
    *,
    structured_root: Path,
    nlg_root: Path,
    label_snapshot: Path,
    model_path: Path,
    run_root: Path,
    selection_file: Path | None = None,
    domains: list[str] | None = None,
) -> list[dict[str, Any]]:
    if phase != "encode" and config.get("status") == "p0_beam_selection_pending":
        raise ValueError(
            "formal manifests require the art/valid beam P0 decision to be frozen"
        )
    configured_domains = list(config["domains"])
    if domains is None:
        domains = configured_domains
    else:
        if len(domains) != len(set(domains)):
            raise ValueError("domains must not contain duplicates")
        unknown_domains = sorted(set(domains) - set(configured_domains))
        if unknown_domains:
            raise ValueError(
                "domains are not present in the frozen config: "
                + ", ".join(unknown_domains)
            )
        if not domains:
            raise ValueError("at least one domain is required")
    seeds = config["training"]["seeds"]
    lambdas = config["selection"]["lambda_grid"]
    jobs = []
    if phase == "encode":
        for domain in domains:
            output = run_root / "embeddings" / f"{domain}.pt"
            jobs.append(
                {
                    "job_id": f"encode-{domain}",
                    "output_dir": str(run_root / "queue" / "encode" / domain),
                    "commands": [[
                        "{python}", "-m", PIPELINE_MODULE, "encode",
                        *_common_data(structured_root, nlg_root, label_snapshot, domain),
                        "--model-path", str(model_path),
                        "--batch-size", "4",
                        "--output", str(output),
                    ]],
                    "expected_files": [str(output), str(output) + ".report.json"],
                }
            )
        return jobs

    if phase == "prepare":
        for domain in domains:
            embedding = run_root / "embeddings" / f"{domain}.pt"
            valid = run_root / "prepared" / "eval" / domain / "valid.pt"
            jobs.append(
                {
                    "job_id": f"prepare-valid-{domain}",
                    "output_dir": str(run_root / "queue" / "prepare" / domain / "valid"),
                    "commands": [[
                        "{python}", "-m", PIPELINE_MODULE, "prepare-eval",
                        *_common_data(structured_root, nlg_root, label_snapshot, domain),
                        "--embeddings", str(embedding),
                        "--split", "valid",
                        "--beam-width", str(config["retrieval"]["beam_width"]),
                        "--output", str(valid),
                    ]],
                    "expected_files": [str(valid), str(valid) + ".report.json"],
                }
            )
            for seed in seeds:
                output = run_root / "prepared" / "train" / domain / f"seed_{seed}.pt"
                jobs.append(
                    {
                        "job_id": f"prepare-train-{domain}-s{seed}",
                        "output_dir": str(
                            run_root / "queue" / "prepare" / domain / f"seed_{seed}"
                        ),
                        "commands": [[
                            "{python}", "-m", PIPELINE_MODULE, "prepare-train",
                            *_common_data(structured_root, nlg_root, label_snapshot, domain),
                            "--embeddings", str(embedding),
                            "--seed", str(seed),
                            "--output", str(output),
                        ]],
                        "expected_files": [str(output), str(output) + ".report.json"],
                    }
                )
        return jobs

    if phase == "sweep":
        for domain in domains:
            embedding = run_root / "embeddings" / f"{domain}.pt"
            valid = run_root / "prepared" / "eval" / domain / "valid.pt"
            for lambda_ in lambdas:
                token = _lambda_token(float(lambda_))
                for seed in seeds:
                    output_dir = (
                        run_root / "sweep" / domain / f"lambda_{token}" / f"seed_{seed}"
                    )
                    training = (
                        run_root / "prepared" / "train" / domain / f"seed_{seed}.pt"
                    )
                    checkpoint = output_dir / "checkpoint.pt"
                    scores = output_dir / "valid_scores.pt"
                    jobs.append(
                        {
                            "job_id": f"sweep-{domain}-l{token}-s{seed}",
                            "output_dir": str(output_dir),
                            "commands": [
                                [
                                    "{python}", "-m", PIPELINE_MODULE, "train",
                                    "--training-data", str(training),
                                    "--embeddings", str(embedding),
                                    "--method", "ours",
                                    "--lambda", str(lambda_),
                                    "--seed", str(seed),
                                    "--device", "cuda",
                                    "--checkpoint", str(checkpoint),
                                ],
                                [
                                    "{python}", "-m", PIPELINE_MODULE, "evaluate",
                                    "--evaluation-data", str(valid),
                                    "--embeddings", str(embedding),
                                    "--checkpoint", str(checkpoint),
                                    "--device", "cuda",
                                    "--scores", str(scores),
                                ],
                            ],
                            "expected_files": [
                                str(checkpoint),
                                str(checkpoint.with_suffix(".report.json")),
                                str(scores),
                                str(scores.with_suffix(".report.json")),
                            ],
                        }
                    )
        return jobs
    if selection_file is None or not selection_file.is_file():
        raise ValueError(f"{phase} requires an existing frozen selection file")
    selection = json.loads(selection_file.read_text(encoding="utf-8"))

    if phase == "prepare-test":
        for domain in domains:
            embedding = run_root / "embeddings" / f"{domain}.pt"
            output = run_root / "prepared" / "eval" / domain / "test.pt"
            jobs.append(
                {
                    "job_id": f"prepare-test-{domain}",
                    "output_dir": str(run_root / "queue" / "prepare-test" / domain),
                    "commands": [[
                        "{python}", "-m", PIPELINE_MODULE, "prepare-eval",
                        *_common_data(structured_root, nlg_root, label_snapshot, domain),
                        "--embeddings", str(embedding),
                        "--split", "test",
                        "--beam-width", str(config["retrieval"]["beam_width"]),
                        "--selection-file", str(selection_file),
                        "--output", str(output),
                    ]],
                    "expected_files": [str(output), str(output) + ".report.json"],
                }
            )
        return jobs

    if phase == "main-test":
        for domain in domains:
            selected_lambda = float(selection["domains"][domain]["lambda"])
            embedding = run_root / "embeddings" / f"{domain}.pt"
            test_data = run_root / "prepared" / "eval" / domain / "test.pt"
            for seed in seeds:
                training = run_root / "prepared" / "train" / domain / f"seed_{seed}.pt"
                sources = {
                    "baseline": run_root / "sweep" / domain / "lambda_1p00" / f"seed_{seed}" / "checkpoint.pt",
                    "ours": run_root / "sweep" / domain / f"lambda_{_lambda_token(selected_lambda)}" / f"seed_{seed}" / "checkpoint.pt",
                }
                for method, checkpoint in sources.items():
                    output_dir = run_root / "test" / domain / method / f"seed_{seed}"
                    scores = output_dir / "test_scores.pt"
                    jobs.append(
                        {
                            "job_id": f"test-{domain}-{method}-s{seed}",
                            "output_dir": str(output_dir),
                            "commands": [[
                                "{python}", "-m", PIPELINE_MODULE, "evaluate",
                                "--evaluation-data", str(test_data),
                                "--embeddings", str(embedding),
                                "--checkpoint", str(checkpoint),
                                "--evaluation-arm", method,
                                "--device", "cuda",
                                "--selection-file", str(selection_file),
                                "--scores", str(scores),
                            ]],
                            "expected_files": [str(scores), str(scores.with_suffix(".report.json"))],
                        }
                    )
                output_dir = run_root / "test" / domain / "matched_random" / f"seed_{seed}"
                checkpoint = output_dir / "checkpoint.pt"
                scores = output_dir / "test_scores.pt"
                jobs.append(
                    {
                        "job_id": f"test-{domain}-matched-random-s{seed}",
                        "output_dir": str(output_dir),
                        "commands": [
                            [
                                "{python}", "-m", PIPELINE_MODULE, "train",
                                "--training-data", str(training),
                                "--embeddings", str(embedding),
                                "--method", "matched_random",
                                "--lambda", str(selected_lambda),
                                "--seed", str(seed),
                                "--device", "cuda",
                                "--checkpoint", str(checkpoint),
                            ],
                            [
                                "{python}", "-m", PIPELINE_MODULE, "evaluate",
                                "--evaluation-data", str(test_data),
                                "--embeddings", str(embedding),
                                "--checkpoint", str(checkpoint),
                                "--evaluation-arm", "matched_random",
                                "--device", "cuda",
                                "--selection-file", str(selection_file),
                                "--scores", str(scores),
                            ],
                        ],
                        "expected_files": [
                            str(checkpoint),
                            str(checkpoint.with_suffix(".report.json")),
                            str(scores),
                            str(scores.with_suffix(".report.json")),
                        ],
                    }
                )
        return jobs

    if phase == "path-sensitivity":
        variants = config["path_selection_sensitivity"]["variant_seeds"]
        for domain in domains:
            selected_lambda = float(selection["domains"][domain]["lambda"])
            embedding = run_root / "embeddings" / f"{domain}.pt"
            test_data = run_root / "prepared" / "eval" / domain / "test.pt"
            for variant in variants:
                for seed in seeds:
                    output_dir = (
                        run_root / "sensitivity" / domain / f"variant_{variant}" / f"seed_{seed}"
                    )
                    training = output_dir / "training.pt"
                    commands = [[
                        "{python}", "-m", PIPELINE_MODULE, "prepare-train",
                        *_common_data(structured_root, nlg_root, label_snapshot, domain),
                        "--embeddings", str(embedding),
                        "--seed", str(seed),
                        "--variant-seed", str(variant),
                        "--output", str(training),
                    ]]
                    expected = [str(training), str(training) + ".report.json"]
                    for method, lambda_ in (("baseline", 1.0), ("ours", selected_lambda)):
                        method_dir = output_dir / method
                        checkpoint = method_dir / "checkpoint.pt"
                        scores = method_dir / "test_scores.pt"
                        commands.extend(
                            [
                                [
                                    "{python}", "-m", PIPELINE_MODULE, "train",
                                    "--training-data", str(training),
                                    "--embeddings", str(embedding),
                                    "--method", method,
                                    "--lambda", str(lambda_),
                                    "--seed", str(seed),
                                    "--device", "cuda",
                                    "--checkpoint", str(checkpoint),
                                ],
                                [
                                    "{python}", "-m", PIPELINE_MODULE, "evaluate",
                                    "--evaluation-data", str(test_data),
                                    "--embeddings", str(embedding),
                                    "--checkpoint", str(checkpoint),
                                    "--evaluation-arm", method,
                                    "--device", "cuda",
                                    "--selection-file", str(selection_file),
                                    "--multiple-shortest-only",
                                    "--scores", str(scores),
                                ],
                            ]
                        )
                        expected.extend(
                            [
                                str(checkpoint),
                                str(checkpoint.with_suffix(".report.json")),
                                str(scores),
                                str(scores.with_suffix(".report.json")),
                            ]
                        )
                    jobs.append(
                        {
                            "job_id": f"sensitivity-{domain}-v{variant}-s{seed}",
                            "output_dir": str(output_dir),
                            "commands": commands,
                            "expected_files": expected,
                        }
                    )
        return jobs
    if phase == "path-sensitivity-prepare":
        variants = config["path_selection_sensitivity"]["variant_seeds"]
        for domain in domains:
            embedding = run_root / "embeddings" / f"{domain}.pt"
            for variant in variants:
                output_dir = (
                    run_root / "sensitivity" / domain / f"variant_{variant}"
                )
                outputs = [
                    output_dir / f"seed_{seed}" / "training.pt"
                    for seed in seeds
                ]
                report = output_dir / "prepare_batch.report.json"
                jobs.append(
                    {
                        "job_id": f"sensitivity-prepare-{domain}-v{variant}",
                        "output_dir": str(output_dir / "prepare"),
                        "commands": [[
                            "{python}", "-m", PREPARE_BATCH_MODULE,
                            "--structured-root", str(structured_root),
                            "--nlg-root", str(nlg_root),
                            "--label-snapshot", str(label_snapshot),
                            "--domain", domain,
                            "--embeddings", str(embedding),
                            "--variant-seed", str(variant),
                            "--seeds", *(str(seed) for seed in seeds),
                            "--outputs", *(str(output) for output in outputs),
                            "--device", "cuda",
                            "--report", str(report),
                        ]],
                        "expected_files": [
                            *(
                                value
                                for output in outputs
                                for value in (str(output), str(output) + ".report.json")
                            ),
                            str(report),
                        ],
                    }
                )
        return jobs
    if phase == "path-sensitivity-train":
        variants = config["path_selection_sensitivity"]["variant_seeds"]
        for domain in domains:
            selected_lambda = float(selection["domains"][domain]["lambda"])
            embedding = run_root / "embeddings" / f"{domain}.pt"
            test_data = run_root / "prepared" / "eval" / domain / "test.pt"
            for variant in variants:
                for seed in seeds:
                    seed_dir = (
                        run_root
                        / "sensitivity"
                        / domain
                        / f"variant_{variant}"
                        / f"seed_{seed}"
                    )
                    training = seed_dir / "training.pt"
                    for method, lambda_ in (
                        ("baseline", 1.0),
                        ("ours", selected_lambda),
                    ):
                        method_dir = seed_dir / method
                        checkpoint = method_dir / "checkpoint.pt"
                        scores = method_dir / "test_scores.pt"
                        jobs.append(
                            {
                                "job_id": (
                                    f"sensitivity-train-{domain}-v{variant}-"
                                    f"s{seed}-{method}"
                                ),
                                "output_dir": str(method_dir),
                                "commands": [
                                    [
                                        "{python}", "-m", PIPELINE_MODULE, "train",
                                        "--training-data", str(training),
                                        "--embeddings", str(embedding),
                                        "--method", method,
                                        "--lambda", str(lambda_),
                                        "--seed", str(seed),
                                        "--device", "cuda",
                                        "--checkpoint", str(checkpoint),
                                    ],
                                    [
                                        "{python}", "-m", PIPELINE_MODULE, "evaluate",
                                        "--evaluation-data", str(test_data),
                                        "--embeddings", str(embedding),
                                        "--checkpoint", str(checkpoint),
                                        "--evaluation-arm", method,
                                        "--device", "cuda",
                                        "--selection-file", str(selection_file),
                                        "--multiple-shortest-only",
                                        "--scores", str(scores),
                                    ],
                                ],
                                "expected_files": [
                                    str(checkpoint),
                                    str(checkpoint.with_suffix(".report.json")),
                                    str(scores),
                                    str(scores.with_suffix(".report.json")),
                                ],
                            }
                        )
        return jobs
    raise ValueError(f"unsupported phase: {phase}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "encode",
            "prepare",
            "sweep",
            "prepare-test",
            "main-test",
            "path-sensitivity",
            "path-sensitivity-prepare",
            "path-sensitivity-train",
        ),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--structured-root", type=Path, required=True)
    parser.add_argument("--nlg-root", type=Path, required=True)
    parser.add_argument("--label-snapshot", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument(
        "--domains",
        nargs="+",
        help="optional ordered subset of domains from the frozen config",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    jobs = build_jobs(
        args.phase,
        config,
        structured_root=args.structured_root,
        nlg_root=args.nlg_root,
        label_snapshot=args.label_snapshot,
        model_path=args.model_path,
        run_root=args.run_root,
        selection_file=args.selection_file,
        domains=args.domains,
    )
    payload = {
        "schema_version": 1,
        "phase": args.phase,
        "config": str(args.config),
        "config_snapshot": config,
        "domain_subset": args.domains or list(config["domains"]),
        "job_count": len(jobs),
        "jobs": jobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{args.phase}: {len(jobs)} jobs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
