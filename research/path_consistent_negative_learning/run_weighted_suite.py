"""Run weighted strategy-2 experiments with at most one process per GPU."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from decimal import Decimal
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
from typing import Sequence

from .weighted_protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
    phase_protocol,
    validate_confirmation_request,
    validate_selection_request,
)


def weight_token(weight: float) -> str:
    return format(Decimal(str(weight)).normalize(), "f").replace(".", "p")


@dataclass(frozen=True)
class WeightedTask:
    strategy: str
    seed: int
    data_path: Path
    output_dir: Path
    disputed_negative_weight: float | None


def build_weighted_tasks(
    data_dir: Path,
    output_dir: Path,
    seeds: Sequence[int],
    *,
    strategy2_weights: Sequence[float],
    random_control_weights: Sequence[float],
    include_baseline: bool = True,
    include_endpoint_reference: bool = False,
) -> tuple[WeightedTask, ...]:
    if len(set(seeds)) != len(seeds):
        raise ValueError("随机种子不得重复")
    if len(set(strategy2_weights)) != len(strategy2_weights):
        raise ValueError("策略2权重不得重复")
    if len(set(random_control_weights)) != len(random_control_weights):
        raise ValueError("随机对照权重不得重复")
    for weight in (*strategy2_weights, *random_control_weights):
        if not 0.0 <= weight <= 1.0:
            raise ValueError("所有权重都必须位于 [0, 1]")
    configurations: list[tuple[str, float | None, str]] = []
    if include_baseline:
        configurations.append(("strategy1_negative", None, "strategy1"))
    if include_endpoint_reference:
        configurations.append(("strategy2_ignore", None, "strategy2_original"))
    configurations.extend(
        (
            "strategy2_weighted",
            float(weight),
            f"strategy2_weight_{weight_token(float(weight))}",
        )
        for weight in strategy2_weights
    )
    configurations.extend(
        (
            "random_weighted",
            float(weight),
            f"random_weight_{weight_token(float(weight))}",
        )
        for weight in random_control_weights
    )
    if len({name for _, _, name in configurations}) != len(configurations):
        raise ValueError("实验配置重复")

    tasks = []
    for seed in seeds:
        data_path = data_dir / f"seed_{seed}.pt"
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        for strategy, weight, directory_name in configurations:
            tasks.append(
                WeightedTask(
                    strategy=strategy,
                    seed=int(seed),
                    data_path=data_path,
                    output_dir=output_dir / directory_name / f"seed_{seed}",
                    disputed_negative_weight=weight,
                )
            )
    return tuple(tasks)


def run_weighted_suite(
    *,
    phase: str,
    data_dir: Path,
    output_dir: Path,
    seeds: Sequence[int],
    gpu_ids: Sequence[int],
    strategy2_weights: Sequence[float],
    random_control_weights: Sequence[float],
    include_baseline: bool,
    include_endpoint_reference: bool,
    evaluation_split: str,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    locked_weight: float | None = None,
) -> None:
    unique_gpus = tuple(dict.fromkeys(int(value) for value in gpu_ids))
    if not unique_gpus:
        raise ValueError("至少需要一张 GPU")
    if len(unique_gpus) > 6:
        raise ValueError("本研究最多使用 6 张 GPU")
    if evaluation_split not in {"selection", "test"}:
        raise ValueError("加权实验只允许在 selection 或 test 划分上汇报")
    protocol = load_protocol(protocol_path)
    settings = phase_protocol(protocol, phase)
    training_config = {
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    if phase == "selection":
        validate_selection_request(
            protocol,
            data_dir=data_dir,
            seeds=seeds,
            strategy2_weights=strategy2_weights,
            random_control_weights=random_control_weights,
            include_baseline=include_baseline,
            include_endpoint_reference=include_endpoint_reference,
            evaluation_split=evaluation_split,
            training_config=training_config,
        )
    elif phase == "confirmation":
        if locked_weight is None:
            raise ValueError("独立确认必须提供选参阶段锁定的权重")
        validate_confirmation_request(
            protocol,
            data_dir=data_dir,
            seeds=seeds,
            locked_weight=locked_weight,
            strategy2_weights=strategy2_weights,
            random_control_weights=random_control_weights,
            include_baseline=include_baseline,
            include_endpoint_reference=include_endpoint_reference,
            evaluation_split=evaluation_split,
            training_config=training_config,
        )
    else:
        raise ValueError(f"该调度器暂不接受阶段：{phase}")
    tasks = build_weighted_tasks(
        data_dir,
        output_dir,
        seeds,
        strategy2_weights=strategy2_weights,
        random_control_weights=random_control_weights,
        include_baseline=include_baseline,
        include_endpoint_reference=include_endpoint_reference,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "COMPLETE").unlink(missing_ok=True)
    manifest = {
        "schema_version": 1,
        "phase": phase,
        "protocol_path": str(protocol_path),
        "locked_weight": locked_weight,
        "seeds": list(seeds),
        "gpu_ids": list(unique_gpus),
        "strategy2_weights": list(strategy2_weights),
        "random_control_weights": list(random_control_weights),
        "include_baseline": include_baseline,
        "include_endpoint_reference": include_endpoint_reference,
        "evaluation_split": evaluation_split,
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "tasks": [
            {
                **asdict(task),
                "data_path": str(task.data_path),
                "output_dir": str(task.output_dir),
            }
            for task in tasks
        ],
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    available_gpus: queue.Queue[int] = queue.Queue()
    for gpu_id in unique_gpus:
        available_gpus.put(gpu_id)

    def run_task(task: WeightedTask) -> tuple[int, str]:
        result_path = task.output_dir / "result.json"
        if result_path.is_file():
            with result_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            expected = {
                "experiment": f"weighted_strategy2_{phase}",
                "domain": settings["domain"],
                "strategy": task.strategy,
                "seed": task.seed,
                "sampler_seed": task.seed,
                "split_seed": settings["split_seed"],
                "split_scheme": settings["split_scheme"],
                "disputed_negative_weight": task.disputed_negative_weight,
                "evaluation_split": evaluation_split,
                "training_config": {
                    "epochs": epochs,
                    "patience": patience,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                },
            }
            mismatches = {
                key: (existing.get(key), value)
                for key, value in expected.items()
                if existing.get(key) != value
            }
            if mismatches:
                raise RuntimeError(
                    f"已有结果与任务不一致：{result_path}: {mismatches}"
                )
            if not result_path.with_name("query_metrics.jsonl").is_file():
                raise RuntimeError(f"已有结果缺少逐问题指标：{result_path}")
            return -1, "skipped"
        gpu_id = available_gpus.get()
        try:
            task.output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "research.path_consistent_negative_learning.train_structured",
                "--data",
                str(task.data_path),
                "--strategy",
                task.strategy,
                "--seed",
                str(task.seed),
                "--output-dir",
                str(task.output_dir),
                "--device",
                "cuda",
                "--epochs",
                str(epochs),
                "--patience",
                str(patience),
                "--batch-size",
                str(batch_size),
                "--learning-rate",
                str(learning_rate),
                "--evaluation-split",
                evaluation_split,
                "--experiment-name",
                f"weighted_strategy2_{phase}",
            ]
            if task.disputed_negative_weight is not None:
                command.extend(
                    [
                        "--disputed-negative-weight",
                        str(task.disputed_negative_weight),
                    ]
                )
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            with (task.output_dir / "stdout.log").open(
                "w", encoding="utf-8"
            ) as stdout, (task.output_dir / "stderr.log").open(
                "w", encoding="utf-8"
            ) as stderr:
                completed = subprocess.run(
                    command,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            if completed.returncode:
                raise RuntimeError(
                    f"训练失败：{task.strategy}, weight="
                    f"{task.disputed_negative_weight}, seed={task.seed}, gpu={gpu_id}"
                )
            return gpu_id, "completed"
        finally:
            available_gpus.put(gpu_id)

    failures = []
    with ThreadPoolExecutor(max_workers=len(unique_gpus)) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                gpu_id, status = future.result()
                print(
                    f"{status}: strategy={task.strategy} "
                    f"weight={task.disputed_negative_weight} "
                    f"seed={task.seed} gpu={gpu_id}",
                    flush=True,
                )
            except Exception as exc:
                failures.append(str(exc))
                print(f"failed: {exc}", flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))
    (output_dir / "COMPLETE").write_text("all tasks completed\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--locked-weight-file", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--strategy2-weights", type=float, nargs="*", default=[])
    parser.add_argument("--random-control-weights", type=float, nargs="*", default=[])
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--include-endpoint-reference", action="store_true")
    parser.add_argument(
        "--evaluation-split",
        choices=("selection", "test"),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    locked_weight = None
    if args.locked_weight_file is not None:
        with args.locked_weight_file.open("r", encoding="utf-8") as handle:
            locked_weight = float(json.load(handle)["selected_weight"])
    run_weighted_suite(
        phase=args.phase,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        gpu_ids=args.gpus,
        strategy2_weights=args.strategy2_weights,
        random_control_weights=args.random_control_weights,
        include_baseline=not args.no_baseline,
        include_endpoint_reference=args.include_endpoint_reference,
        evaluation_split=args.evaluation_split,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        protocol_path=args.protocol,
        locked_weight=locked_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
