"""Run independent Retriever-only jobs on at most six specified GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
from typing import Any, Sequence


ALLOWED_GPUS = frozenset(range(6))


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _available_gpus(gpus: Sequence[int], maximum_used_mib: int) -> tuple[int, ...]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    used = {
        int(index.strip()): int(memory.strip())
        for index, memory in (line.split(",") for line in output.splitlines())
    }
    return tuple(gpu for gpu in gpus if used.get(gpu, maximum_used_mib + 1) <= maximum_used_mib)


def _wait_for_gpus(
    gpus: Sequence[int],
    maximum_used_mib: int,
    *,
    timeout_seconds: float,
) -> tuple[int, ...]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        available = _available_gpus(gpus, maximum_used_mib)
        if available:
            return available
        if time.monotonic() >= deadline:
            return ()
        time.sleep(5.0)


def run_manifest(
    manifest_path: Path,
    *,
    gpus: Sequence[int],
    maximum_used_mib: int,
) -> None:
    requested = tuple(dict.fromkeys(gpus))
    if not requested or len(requested) > 6 or not set(requested) <= ALLOWED_GPUS:
        raise ValueError("GPU IDs must be a nonempty subset of 0--5")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest["jobs"]
    job_ids = [job["job_id"] for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("job IDs must be unique")
    available = _wait_for_gpus(requested, maximum_used_mib, timeout_seconds=120.0)
    if not available:
        raise RuntimeError("none of the requested GPUs became available")

    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in available:
        gpu_pool.put(gpu)

    def execute(job: dict[str, Any]) -> tuple[str, int]:
        expected = [Path(value) for value in job["expected_files"]]
        if expected and all(path.is_file() for path in expected):
            return "skipped", -1
        gpu = gpu_pool.get()
        try:
            output_dir = Path(job["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["TRANSFORMERS_OFFLINE"] = "1"
            environment["HF_HUB_OFFLINE"] = "1"
            commands = [
                [sys.executable if token == "{python}" else str(token) for token in command]
                for command in job["commands"]
            ]
            (output_dir / "commands.json").write_text(
                json.dumps(commands, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            started = _timestamp()
            for command_index, command in enumerate(commands):
                with (output_dir / f"command_{command_index}.stdout.log").open(
                    "w", encoding="utf-8"
                ) as stdout, (output_dir / f"command_{command_index}.stderr.log").open(
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
                        f"{job['job_id']} command {command_index} failed with "
                        f"exit code {completed.returncode}"
                    )
            missing = [str(path) for path in expected if not path.is_file()]
            if missing:
                raise RuntimeError(f"{job['job_id']} did not produce: {missing}")
            (output_dir / "queue_provenance.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "job_id": job["job_id"],
                        "phase": manifest["phase"],
                        "started_at": started,
                        "completed_at": _timestamp(),
                        "physical_gpu_id": gpu,
                        "commands": commands,
                        "expected_files": [str(path) for path in expected],
                        "manifest": str(manifest_path.resolve()),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
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
                print(f"{status}: {job['job_id']} gpu={gpu}", flush=True)
            except Exception as exc:
                failures.append({"job_id": job["job_id"], "error": str(exc)})
                print(f"failed: {job['job_id']}: {exc}", flush=True)
    failure_path = manifest_path.with_suffix(".failures.json")
    if failures:
        failure_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"{len(failures)} Retriever-only jobs failed")
    failure_path.unlink(missing_ok=True)
    manifest_path.with_suffix(".complete").write_text(
        "all jobs completed\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--maximum-used-mib", type=int, default=512)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_manifest(
        args.manifest,
        gpus=args.gpus,
        maximum_used_mib=args.maximum_used_mib,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
