"""Load and enforce the frozen weighted strategy-2 experiment protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .structured_data import StructuredExperimentData


DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parent / "protocols" / "weighted_strategy2.json"
)


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def phase_protocol(
    protocol: Mapping[str, Any],
    phase: str,
) -> Mapping[str, Any]:
    key = {
        "selection": "selection",
        "confirmation": "independent_confirmation",
        "gate_d": "gate_d",
    }.get(phase)
    if key is None:
        raise ValueError(f"未知加权实验阶段：{phase}")
    return protocol[key]


def validate_dataset_files(
    data_dir: Path,
    seeds: Sequence[int],
    *,
    expected_domain: str,
    expected_split_seed: int,
    expected_split_scheme: str,
) -> None:
    for seed in seeds:
        path = data_dir / f"seed_{seed}.pt"
        data = StructuredExperimentData.load(path)
        actual = (
            data.domain,
            data.sampler_seed,
            data.split_seed,
            data.split_scheme,
        )
        expected = (
            expected_domain,
            seed,
            expected_split_seed,
            expected_split_scheme,
        )
        if actual != expected:
            raise ValueError(f"数据文件不符合冻结协议：{path}: {actual} != {expected}")


def validate_selection_request(
    protocol: Mapping[str, Any],
    *,
    data_dir: Path,
    seeds: Sequence[int],
    strategy2_weights: Sequence[float],
    random_control_weights: Sequence[float],
    include_baseline: bool,
    include_endpoint_reference: bool,
    evaluation_split: str,
    training_config: Mapping[str, Any],
) -> None:
    settings = phase_protocol(protocol, "selection")
    candidate_weights = [float(value) for value in settings["candidate_weights"]]
    diagnostic_weights = [
        float(value) for value in settings["diagnostic_endpoint_weights"]
    ]
    expected_strategy2 = sorted(candidate_weights + diagnostic_weights)
    if list(seeds) != list(settings["sampler_and_training_seeds"]):
        raise ValueError("选参随机种子不符合冻结协议")
    if sorted(strategy2_weights) != expected_strategy2:
        raise ValueError("策略2选参权重网格不符合冻结协议")
    if sorted(random_control_weights) != sorted(candidate_weights):
        raise ValueError("随机降权对照网格不符合冻结协议")
    if not include_baseline or not include_endpoint_reference:
        raise ValueError("选参阶段必须包含策略1和原策略2端点参照")
    if evaluation_split != settings["reported_split"]:
        raise ValueError("选参评估划分不符合冻结协议")
    if dict(training_config) != dict(settings["training"]):
        raise ValueError("选参训练超参数不符合冻结协议")
    validate_dataset_files(
        data_dir,
        seeds,
        expected_domain=str(settings["domain"]),
        expected_split_seed=int(settings["split_seed"]),
        expected_split_scheme=str(settings["split_scheme"]),
    )


def validate_confirmation_request(
    protocol: Mapping[str, Any],
    *,
    data_dir: Path,
    seeds: Sequence[int],
    locked_weight: float,
    strategy2_weights: Sequence[float],
    random_control_weights: Sequence[float],
    include_baseline: bool,
    include_endpoint_reference: bool,
    evaluation_split: str,
    training_config: Mapping[str, Any],
) -> None:
    settings = phase_protocol(protocol, "confirmation")
    candidate_weights = [
        float(value) for value in phase_protocol(protocol, "selection")["candidate_weights"]
    ]
    if locked_weight not in candidate_weights:
        raise ValueError("独立确认权重不属于冻结的候选网格")
    if list(seeds) != list(settings["sampler_and_training_seeds"]):
        raise ValueError("独立确认随机种子不符合冻结协议")
    if list(strategy2_weights) != [locked_weight]:
        raise ValueError("独立确认只能运行锁定的策略2权重")
    if list(random_control_weights) != [locked_weight]:
        raise ValueError("独立确认只能运行锁定的随机对照权重")
    if not include_baseline or include_endpoint_reference:
        raise ValueError("独立确认的实验臂不符合冻结协议")
    if evaluation_split != settings["reported_split"]:
        raise ValueError("独立确认评估划分不符合冻结协议")
    if dict(training_config) != dict(settings["training"]):
        raise ValueError("独立确认训练超参数不符合冻结协议")
    validate_dataset_files(
        data_dir,
        seeds,
        expected_domain=str(settings["domain"]),
        expected_split_seed=int(settings["split_seed"]),
        expected_split_scheme=str(settings["split_scheme"]),
    )
