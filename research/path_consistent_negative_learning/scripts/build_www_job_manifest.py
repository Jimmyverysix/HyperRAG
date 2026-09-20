"""按冻结协议生成 selection、test-checkpoint 或 test-control GPU 任务。"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
from typing import Any


def _token(value: float) -> str:
    return format(Decimal(str(value)).normalize(), "f").replace(".", "p")


def _training_args(config: dict[str, Any]) -> list[str]:
    training = config["training"]
    args = [
        "--epochs",
        str(training["epochs"]),
        "--patience",
        str(training["patience"]),
        "--batch-size",
        str(training["batch_size"]),
        "--learning-rate",
        str(training["learning_rate"]),
    ]
    if training.get("amp", False):
        args.append("--amp")
    return args


def _selection_jobs(
    config: dict[str, Any], data_root: Path, run_root: Path
) -> list[dict[str, Any]]:
    jobs = []
    split_seed = config["randomness"]["split_seed"]
    split_scheme = config["randomness"]["split_scheme"]
    for domain in config["datasets"]:
        for lambda_value in config["lambda_grid"]:
            arm = f"strategy2_weight_{_token(lambda_value)}"
            for seed in config["randomness"]["sampler_and_training_seeds"]:
                data = data_root / domain / f"seed_{seed}.pt"
                output = run_root / "selection" / domain / arm / f"seed_{seed}"
                command = [
                    "{python}",
                    "-m",
                    "research.path_consistent_negative_learning.train_structured",
                    "--data",
                    str(data),
                    "--strategy",
                    "strategy2_weighted",
                    "--disputed-negative-weight",
                    str(lambda_value),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cuda",
                    "--evaluation-split",
                    "selection",
                    "--experiment-name",
                    "www_revision_proxy_selection",
                    *_training_args(config),
                ]
                jobs.append(
                    {
                        "experiment_id": f"selection-{domain}-l{_token(lambda_value)}-s{seed}",
                        "command": command,
                        "output_dir": str(output),
                        "expected_result": {
                            "experiment": "www_revision_proxy_selection",
                            "domain": domain,
                            "strategy": "strategy2_weighted",
                            "seed": seed,
                            "sampler_seed": seed,
                            "split_seed": split_seed,
                            "split_scheme": split_scheme,
                            "evaluation_split": "selection",
                            "disputed_negative_weight": lambda_value,
                            "amp": bool(config["training"].get("amp", False)),
                        },
                    }
                )
    return jobs


def _checkpoint_test_jobs(
    config: dict[str, Any], data_root: Path, run_root: Path, lambda_root: Path
) -> list[dict[str, Any]]:
    jobs = []
    for domain in config["datasets"]:
        lambda_file = lambda_root / domain / "lambda_star.json"
        selection = json.loads(lambda_file.read_text(encoding="utf-8"))
        best = float(selection["best_lambda"])
        arms = {
            "baseline_lambda_1": 1.0,
            "mask_endpoint_lambda_0": 0.0,
            "dataset_specific_lambda_star": best,
        }
        for arm, lambda_value in arms.items():
            source_arm = f"strategy2_weight_{_token(lambda_value)}"
            for seed in config["randomness"]["sampler_and_training_seeds"]:
                data = data_root / domain / f"seed_{seed}.pt"
                source = (
                    run_root
                    / "selection"
                    / domain
                    / source_arm
                    / f"seed_{seed}"
                    / "result.json"
                )
                if not source.is_file():
                    raise FileNotFoundError(source)
                output = run_root / "test" / domain / arm / f"seed_{seed}"
                jobs.append(
                    {
                        "experiment_id": f"test-{domain}-{arm}-s{seed}",
                        "command": [
                            "{python}",
                            "-m",
                            "research.path_consistent_negative_learning.scripts.evaluate_checkpoint",
                            "--data",
                            str(data),
                            "--source-result",
                            str(source),
                            "--lambda-star",
                            str(lambda_file),
                            "--arm",
                            arm,
                            "--output-dir",
                            str(output),
                            "--device",
                            "cuda",
                            "--batch-size",
                            str(config["training"]["batch_size"]),
                        ],
                        "output_dir": str(output),
                        "expected_result": {
                            "experiment": "www_revision_proxy_test",
                            "domain": domain,
                            "arm": arm,
                            "seed": seed,
                            "evaluation_split": "test",
                            "disputed_negative_weight": lambda_value,
                        },
                    }
                )
    return jobs


def _control_test_jobs(
    config: dict[str, Any], data_root: Path, run_root: Path, lambda_root: Path
) -> list[dict[str, Any]]:
    jobs = []
    for domain in config["datasets"]:
        lambda_file = lambda_root / domain / "lambda_star.json"
        selection = json.loads(lambda_file.read_text(encoding="utf-8"))
        best = float(selection["best_lambda"])
        for arm, strategy, lambda_value in (
            ("matched_random_at_lambda_star", "random_weighted", best),
            ("all_shortest_positive", "strategy3_positive", None),
        ):
            for seed in config["randomness"]["sampler_and_training_seeds"]:
                data = data_root / domain / f"seed_{seed}.pt"
                output = run_root / "test" / domain / arm / f"seed_{seed}"
                command = [
                    "{python}",
                    "-m",
                    "research.path_consistent_negative_learning.train_structured",
                    "--data",
                    str(data),
                    "--strategy",
                    strategy,
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cuda",
                    "--evaluation-split",
                    "test",
                    "--experiment-name",
                    "www_revision_proxy_test_control",
                    *_training_args(config),
                ]
                if lambda_value is not None:
                    command.extend(["--disputed-negative-weight", str(lambda_value)])
                jobs.append(
                    {
                        "experiment_id": f"control-{domain}-{arm}-s{seed}",
                        "command": command,
                        "output_dir": str(output),
                        "expected_result": {
                            "experiment": "www_revision_proxy_test_control",
                            "domain": domain,
                            "strategy": strategy,
                            "seed": seed,
                            "evaluation_split": "test",
                            "disputed_negative_weight": lambda_value,
                        },
                    }
                )
    return jobs


def build_jobs(
    phase: str,
    config: dict[str, Any],
    data_root: Path,
    run_root: Path,
    lambda_root: Path | None,
) -> list[dict[str, Any]]:
    if phase == "selection":
        return _selection_jobs(config, data_root, run_root)
    if lambda_root is None:
        raise ValueError(f"{phase} 需要 --lambda-root")
    if phase == "test-checkpoints":
        return _checkpoint_test_jobs(config, data_root, run_root, lambda_root)
    if phase == "test-controls":
        return _control_test_jobs(config, data_root, run_root, lambda_root)
    raise ValueError(f"未知 phase：{phase}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("selection", "test-checkpoints", "test-controls"), required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lambda-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    jobs = build_jobs(
        args.phase, config, args.data_root, args.run_root, args.lambda_root
    )
    manifest = {
        "schema_version": 1,
        "phase": args.phase,
        "config": str(args.config),
        "job_count": len(jobs),
        "jobs": jobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{args.phase}: {len(jobs)} jobs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
