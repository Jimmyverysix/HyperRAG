"""Run the frozen Gate-D suites sequentially after confirmation passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .prepare_gate_d import load_gate_d_settings
from .run_weighted_suite import run_weighted_suite
from .weighted_protocol import DEFAULT_PROTOCOL_PATH, phase_protocol, validate_dataset_files


ALLOWED_GPU_IDS = frozenset(range(6))


def read_passing_confirmation(
    path: Path,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    confirmation = phase_protocol(protocol, "confirmation")
    actual = (
        summary.get("experiment"),
        summary.get("domain"),
        summary.get("evaluation_split"),
    )
    expected = (
        "weighted_strategy2_independent_confirmation",
        confirmation["domain"],
        confirmation["reported_split"],
    )
    if actual != expected:
        raise ValueError(f"独立确认摘要与冻结协议不一致：{actual} != {expected}")
    if summary.get("gate", {}).get("proceed_to_gate_d") is not True:
        raise RuntimeError("独立确认没有通过，禁止启动门槛 D")
    if summary.get("locked_weight") is None:
        raise ValueError("独立确认摘要缺少锁定权重")
    locked_weight = float(summary["locked_weight"])
    candidates = {
        float(value) for value in phase_protocol(protocol, "selection")["candidate_weights"]
    }
    if locked_weight not in candidates:
        raise ValueError("独立确认的锁定权重不属于冻结候选网格")
    return summary, locked_weight


def _write_or_validate_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != manifest:
            raise RuntimeError(f"已有门槛 D 清单与本次冻结配置不一致：{path}")
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run_all_gate_d(
    data_root: Path,
    output_root: Path,
    confirmation_summary: Path,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    gpu_ids: Sequence[int] = (0, 1, 2, 3, 4, 5),
    suite_runner: Callable[..., None] = run_weighted_suite,
) -> dict[str, object]:
    protocol, settings = load_gate_d_settings(protocol_path)
    unique_gpus = tuple(dict.fromkeys(int(value) for value in gpu_ids))
    if tuple(int(value) for value in gpu_ids) != unique_gpus:
        raise ValueError("GPU 编号不得重复")
    if not unique_gpus or not set(unique_gpus).issubset(ALLOWED_GPU_IDS):
        raise ValueError("门槛 D 只能使用 GPU 0 到 GPU 5")

    _, locked_weight = read_passing_confirmation(confirmation_summary, protocol)
    domains = tuple(str(value) for value in settings["domains"])
    seeds = tuple(int(value) for value in settings["sampler_and_training_seeds"])
    for domain in domains:
        validate_dataset_files(
            data_root / domain / "datasets",
            seeds,
            expected_domain=domain,
            expected_split_seed=int(settings["split_seed"]),
            expected_split_scheme=str(settings["split_scheme"]),
        )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "phase": "gate_d",
        "protocol_path": str(protocol_path),
        "confirmation_summary": str(confirmation_summary),
        "locked_weight": locked_weight,
        "domains": list(domains),
        "seeds": list(seeds),
        "gpu_ids": list(unique_gpus),
        "evaluation_split": settings["reported_split"],
        "split_seed": settings["split_seed"],
        "split_scheme": settings["split_scheme"],
        "training": settings["training"],
    }
    _write_or_validate_manifest(output_root / "run_manifest.json", manifest)
    complete_path = output_root / "COMPLETE"
    complete_path.unlink(missing_ok=True)

    training = settings["training"]
    completed_domains = []
    for domain in domains:
        domain_output = output_root / domain / "runs"
        suite_runner(
            phase="gate_d",
            data_dir=data_root / domain / "datasets",
            output_dir=domain_output,
            seeds=seeds,
            gpu_ids=unique_gpus,
            strategy2_weights=(locked_weight,),
            random_control_weights=(locked_weight,),
            include_baseline=True,
            include_endpoint_reference=False,
            evaluation_split=str(settings["reported_split"]),
            epochs=int(training["epochs"]),
            patience=int(training["patience"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            protocol_path=protocol_path,
            locked_weight=locked_weight,
            domain=domain,
        )
        if not (domain_output / "COMPLETE").is_file():
            raise RuntimeError(f"领域 {domain} 的调度器没有写入 COMPLETE")
        completed_domains.append(domain)
        print(f"completed domain: {domain}", flush=True)

    complete_path.write_text("all gate-d domains completed\n", encoding="utf-8")
    return {
        "locked_weight": locked_weight,
        "completed_domains": completed_domains,
        "domain_count": len(completed_domains),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--confirmation-summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_all_gate_d(
        args.data_root,
        args.output_root,
        args.confirmation_summary,
        protocol_path=args.protocol,
        gpu_ids=args.gpus,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
