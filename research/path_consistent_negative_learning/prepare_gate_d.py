"""Prepare the frozen Gate-D structured datasets with bounded concurrency."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping, Sequence

from .weighted_protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
    phase_protocol,
    validate_dataset_files,
)


MAX_PREPARATION_PROCESSES = 6
EXPECTED_DOMAIN_COUNT = 11
EXPECTED_SEED_COUNT = 5
FROZEN_STATUS = "frozen_before_weighted_results"


@dataclass(frozen=True)
class GateDPreparationTask:
    domain: str
    seed: int
    domain_dir: Path
    output_path: Path
    split_seed: int
    split_scheme: str


def load_gate_d_settings(
    protocol_path: Path,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    protocol = load_protocol(protocol_path)
    if protocol.get("schema_version") != 1 or protocol.get("status") != FROZEN_STATUS:
        raise ValueError("门槛 D 只能读取结果产生前冻结的第一版加权协议")
    settings = phase_protocol(protocol, "gate_d")
    domains = list(settings["domains"])
    seeds = list(settings["sampler_and_training_seeds"])
    if len(domains) != EXPECTED_DOMAIN_COUNT or len(set(domains)) != len(domains):
        raise ValueError("冻结协议必须包含 11 个互不重复的门槛 D 领域")
    if len(seeds) != EXPECTED_SEED_COUNT or len(set(seeds)) != len(seeds):
        raise ValueError("冻结协议必须包含 5 个互不重复的门槛 D 随机种子")
    if settings.get("run_only_after_independent_confirmation_passes") is not True:
        raise ValueError("冻结协议必须要求独立确认通过后才能运行门槛 D")
    if settings.get("locked_weight_across_domains") is not True:
        raise ValueError("冻结协议必须在所有门槛 D 领域使用同一个锁定权重")
    return protocol, settings


def build_preparation_tasks(
    dataset_root: Path,
    output_root: Path,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> tuple[GateDPreparationTask, ...]:
    _, settings = load_gate_d_settings(protocol_path)
    tasks = []
    for domain in settings["domains"]:
        domain_dir = dataset_root / str(domain)
        if not domain_dir.is_dir():
            raise FileNotFoundError(domain_dir)
        for seed in settings["sampler_and_training_seeds"]:
            tasks.append(
                GateDPreparationTask(
                    domain=str(domain),
                    seed=int(seed),
                    domain_dir=domain_dir,
                    output_path=(
                        output_root / str(domain) / "datasets" / f"seed_{seed}.pt"
                    ),
                    split_seed=int(settings["split_seed"]),
                    split_scheme=str(settings["split_scheme"]),
                )
            )
    return tuple(tasks)


def _validate_task_output(task: GateDPreparationTask) -> None:
    validate_dataset_files(
        task.output_path.parent,
        (task.seed,),
        expected_domain=task.domain,
        expected_split_seed=task.split_seed,
        expected_split_scheme=task.split_scheme,
    )


def _run_preparation_task(task: GateDPreparationTask) -> None:
    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = task.output_path.parent.parent / "prepare_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "research.path_consistent_negative_learning.prepare_structured_experiment",
        "--domain-dir",
        str(task.domain_dir),
        "--sampler-seed",
        str(task.seed),
        "--split-seed",
        str(task.split_seed),
        "--split-scheme",
        task.split_scheme,
        "--output",
        str(task.output_path),
    ]
    with (log_dir / f"seed_{task.seed}.stdout.log").open(
        "w", encoding="utf-8"
    ) as stdout, (log_dir / f"seed_{task.seed}.stderr.log").open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"结构数据准备失败：domain={task.domain}, seed={task.seed}, "
            f"returncode={completed.returncode}"
        )


def _write_or_validate_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != manifest:
            raise RuntimeError(f"已有准备清单与冻结协议不一致：{path}")
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def prepare_gate_d_data(
    dataset_root: Path,
    output_root: Path,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    max_workers: int = MAX_PREPARATION_PROCESSES,
    task_runner: Callable[[GateDPreparationTask], None] = _run_preparation_task,
) -> dict[str, int]:
    if not 1 <= max_workers <= MAX_PREPARATION_PROCESSES:
        raise ValueError("门槛 D 数据准备的并发子进程数必须位于 1 到 6")
    _, settings = load_gate_d_settings(protocol_path)
    tasks = build_preparation_tasks(
        dataset_root,
        output_root,
        protocol_path=protocol_path,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "phase": "gate_d_data_preparation",
        "protocol_path": str(protocol_path),
        "domains": list(settings["domains"]),
        "seeds": list(settings["sampler_and_training_seeds"]),
        "split_seed": settings["split_seed"],
        "split_scheme": settings["split_scheme"],
        "max_workers": max_workers,
        "tasks": [
            {
                **asdict(task),
                "domain_dir": str(task.domain_dir),
                "output_path": str(task.output_path),
            }
            for task in tasks
        ],
    }
    _write_or_validate_manifest(output_root / "prepare_manifest.json", manifest)
    complete_path = output_root / "COMPLETE"
    complete_path.unlink(missing_ok=True)

    pending = []
    skipped = 0
    for task in tasks:
        if task.output_path.exists():
            _validate_task_output(task)
            skipped += 1
        else:
            pending.append(task)

    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task_runner, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
                _validate_task_output(task)
                print(f"prepared: domain={task.domain} seed={task.seed}", flush=True)
            except Exception as exc:
                failures.append(f"domain={task.domain}, seed={task.seed}: {exc}")
                print(f"failed: {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))

    for task in tasks:
        _validate_task_output(task)
    complete_path.write_text("all gate-d datasets prepared\n", encoding="utf-8")
    return {"prepared": len(pending), "skipped": skipped, "total": len(tasks)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--workers", type=int, default=MAX_PREPARATION_PROCESSES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = prepare_gate_d_data(
        args.dataset_root,
        args.output_root,
        protocol_path=args.protocol,
        max_workers=args.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
