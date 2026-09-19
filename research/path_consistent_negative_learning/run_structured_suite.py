"""Run the four Gate C arms across shared seeds with one process per GPU."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
from typing import Sequence

from .strategies import STRATEGIES


@dataclass(frozen=True)
class Task:
    strategy: str
    seed: int
    data_path: Path
    output_dir: Path


def build_tasks(
    data_dir: Path,
    output_dir: Path,
    seeds: Sequence[int],
) -> tuple[Task, ...]:
    tasks = []
    for seed in seeds:
        data_path = data_dir / f"seed_{seed}.pt"
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        for strategy in STRATEGIES:
            tasks.append(
                Task(
                    strategy=strategy,
                    seed=int(seed),
                    data_path=data_path,
                    output_dir=output_dir / strategy / f"seed_{seed}",
                )
            )
    return tuple(tasks)


def run_suite(
    *,
    data_dir: Path,
    output_dir: Path,
    seeds: Sequence[int],
    gpu_ids: Sequence[int],
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    unique_gpus = tuple(dict.fromkeys(int(value) for value in gpu_ids))
    if not unique_gpus:
        raise ValueError("至少需要一张 GPU")
    if len(unique_gpus) > 6:
        raise ValueError("本研究最多使用 6 张 GPU")
    tasks = build_tasks(data_dir, output_dir, seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seeds": list(seeds),
                "gpu_ids": list(unique_gpus),
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
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    available_gpus: queue.Queue[int] = queue.Queue()
    for gpu_id in unique_gpus:
        available_gpus.put(gpu_id)

    def run_task(task: Task) -> tuple[Task, int, str]:
        result_path = task.output_dir / "result.json"
        if result_path.is_file():
            return task, -1, "skipped"
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
            ]
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
                    f"训练失败：{task.strategy}, seed={task.seed}, gpu={gpu_id}"
                )
            return task, gpu_id, "completed"
        finally:
            available_gpus.put(gpu_id)

    failures = []
    with ThreadPoolExecutor(max_workers=len(unique_gpus)) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                _, gpu_id, status = future.result()
                print(
                    f"{status}: strategy={task.strategy} seed={task.seed} gpu={gpu_id}",
                    flush=True,
                )
            except Exception as exc:  # subprocess details are in the task log
                failures.append(str(exc))
                print(f"failed: {exc}", flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))
    (output_dir / "COMPLETE").write_text("all tasks completed\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_suite(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        gpu_ids=args.gpus,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
