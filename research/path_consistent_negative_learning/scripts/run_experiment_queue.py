"""在最多六张指定 GPU 上执行可恢复的独立实验队列。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch


ALLOWED_GPU_IDS = frozenset(range(6))
GPU_RELEASE_TIMEOUT_SECONDS = 120.0
GPU_RELEASE_POLL_SECONDS = 5.0


def _available_gpus(requested: Sequence[int], memory_limit_mib: int) -> tuple[int, ...]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    used = {}
    for line in completed.stdout.splitlines():
        index, memory = (int(value.strip()) for value in line.split(","))
        used[index] = memory
    return tuple(gpu for gpu in requested if used.get(gpu, memory_limit_mib + 1) <= memory_limit_mib)


def _wait_for_available_gpus(
    requested: Sequence[int],
    memory_limit_mib: int,
    *,
    timeout_seconds: float = GPU_RELEASE_TIMEOUT_SECONDS,
    poll_seconds: float = GPU_RELEASE_POLL_SECONDS,
) -> tuple[int, ...]:
    """等待上一批 CUDA 进程释放显存，再确定本阶段可用 GPU。"""

    deadline = time.monotonic() + timeout_seconds
    announced = False
    while True:
        available = _available_gpus(requested, memory_limit_mib)
        if available:
            return available
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ()
        if not announced:
            print("等待上一阶段释放 GPU 显存……", flush=True)
            announced = True
        time.sleep(min(poll_seconds, remaining))


def _matches_expected(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "dirty": bool(status),
        "changed_paths": status,
    }


def _runtime_state() -> dict[str, Any]:
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "nvidia_driver_versions": sorted(set(value.strip() for value in driver)),
    }


def run_queue(
    manifest_path: Path,
    *,
    gpu_ids: Sequence[int],
    memory_limit_mib: int = 512,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "config_snapshot" not in manifest:
        raise ValueError("manifest 缺少冻结的 config_snapshot")
    jobs = manifest["jobs"]
    ids = [job["experiment_id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("experiment_id 必须唯一")

    requested = tuple(dict.fromkeys(int(gpu) for gpu in gpu_ids))
    if not requested or len(requested) > 6 or not set(requested) <= ALLOWED_GPU_IDS:
        raise ValueError("GPU 只能从 0--5 中选择，且并发数不得超过 6")
    available = _wait_for_available_gpus(requested, memory_limit_mib)
    if not available:
        raise RuntimeError("指定 GPU 当前均不空闲")

    root = manifest_path.resolve().parent
    run_environment = {
        "captured_at": _timestamp(),
        "working_directory": str(Path.cwd().resolve()),
        "manifest": str(manifest_path.resolve()),
        "git": _git_state(),
        "runtime": _runtime_state(),
        "config_path": manifest["config"],
        "config_snapshot": manifest["config_snapshot"],
    }
    (root / "environment.json").write_text(
        json.dumps(run_environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "COMPLETE").unlink(missing_ok=True)
    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in available:
        gpu_pool.put(gpu)

    def execute(job: dict[str, Any]) -> tuple[str, int]:
        output_dir = Path(job["output_dir"])
        result_path = output_dir / "result.json"
        provenance_path = output_dir / "provenance.json"
        if result_path.is_file():
            actual = json.loads(result_path.read_text(encoding="utf-8"))
            if not _matches_expected(actual, job["expected_result"]):
                raise RuntimeError(f"已有结果与 manifest 不一致：{result_path}")
            if not (output_dir / "query_metrics.jsonl").is_file():
                raise RuntimeError(f"已有结果缺少逐题指标：{output_dir}")
            if not provenance_path.is_file():
                raise RuntimeError(f"已有结果缺少执行 provenance：{output_dir}")
            return "skipped", -1

        gpu = gpu_pool.get()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable if token == "{python}" else str(token)
                for token in job["command"]
            ]
            (output_dir / "command.json").write_text(
                json.dumps(command, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            started_at = _timestamp()
            with (output_dir / "stdout.log").open("w", encoding="utf-8") as stdout:
                with (output_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
                    completed = subprocess.run(
                        command,
                        env=environment,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                    )
            if completed.returncode:
                raise RuntimeError(
                    f"实验失败：{job['experiment_id']}，详见 {output_dir / 'stderr.log'}"
                )
            actual = json.loads(result_path.read_text(encoding="utf-8"))
            if not _matches_expected(actual, job["expected_result"]):
                raise RuntimeError(f"新结果与 manifest 不一致：{result_path}")
            provenance = {
                "schema_version": 1,
                "experiment_id": job["experiment_id"],
                "phase": manifest["phase"],
                "started_at": started_at,
                "completed_at": _timestamp(),
                "physical_gpu_id": gpu,
                "command": command,
                "expected_result": job["expected_result"],
                "git": run_environment["git"],
                "runtime": run_environment["runtime"],
                "config_path": manifest["config"],
                "config_snapshot": manifest["config_snapshot"],
            }
            provenance_path.write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return "completed", gpu
        finally:
            gpu_pool.put(gpu)

    failures = []
    with ThreadPoolExecutor(max_workers=len(available)) as executor:
        futures = {executor.submit(execute, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                status, gpu = future.result()
                print(f"{status}: {job['experiment_id']} gpu={gpu}", flush=True)
            except Exception as exc:
                failures.append(str(exc))
                print(f"failed: {exc}", flush=True)
    if failures:
        (root / "FAILURES.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"{len(failures)} 个任务失败")
    (root / "FAILURES.json").unlink(missing_ok=True)
    (root / "COMPLETE").write_text("all jobs completed\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--memory-limit-mib", type=int, default=512)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_queue(
        args.manifest,
        gpu_ids=args.gpus,
        memory_limit_mib=args.memory_limit_mib,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
