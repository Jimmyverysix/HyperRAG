"""Evaluate the locked strategy-2 weight on an independent domain test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .run_weighted_suite import weight_token
from .weighted_results import (
    compare_configurations,
    load_weighted_runs,
    summarize_configurations,
    write_json,
)
from .weighted_protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
    phase_protocol,
)


MAX_CONFIRMATION_PRECISION_DROP = 0.01


def _read_locked_weight(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    weight = payload.get("selected_weight")
    if weight is None:
        raise ValueError("选参阶段没有锁定可确认的权重")
    return float(weight)


def confirm_weight(
    run_root: Path,
    *,
    locked_weight: float,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    settings = phase_protocol(protocol, "confirmation")
    candidate_weights = [
        float(value) for value in phase_protocol(protocol, "selection")["candidate_weights"]
    ]
    if locked_weight not in candidate_weights:
        raise ValueError("锁定权重不属于冻结的候选网格")
    runs = load_weighted_runs(run_root, expected_evaluation_split="test")
    strategy2 = f"strategy2_weight_{weight_token(locked_weight)}"
    random_control = f"random_weight_{weight_token(locked_weight)}"
    required = {"strategy1", strategy2, random_control}
    if set(runs) != required:
        raise ValueError(
            "独立确认只能包含锁定权重的策略1、策略2加权版和随机降权对照；"
            f"实际为 {sorted(runs)}"
        )
    expected_seeds = list(settings["sampler_and_training_seeds"])
    for configuration, seed_runs in runs.items():
        if sorted(seed_runs) != expected_seeds:
            raise ValueError(f"配置 {configuration} 的随机种子不符合冻结协议")
        for run in seed_runs.values():
            result = run["result"]
            actual = (
                result["domain"],
                result["split_seed"],
                result["split_scheme"],
                result["training_config"],
                result["experiment"],
            )
            expected = (
                settings["domain"],
                settings["split_seed"],
                settings["split_scheme"],
                settings["training"],
                "weighted_strategy2_confirmation",
            )
            if actual != expected:
                raise ValueError(f"独立确认结果不符合冻结协议：{run['path']}")

    versus_baseline = compare_configurations(
        runs,
        reference="strategy1",
        comparison=strategy2,
        metric="answer_reach_10",
    )
    versus_random = compare_configurations(
        runs,
        reference=random_control,
        comparison=strategy2,
        metric="answer_reach_10",
    )
    precision_noninferiority = compare_configurations(
        runs,
        reference="strategy1",
        comparison=strategy2,
        metric="selected_pr_auc",
        confidence_level=0.90,
    )
    baseline_low = versus_baseline["paired_bootstrap"]["ci_low"]
    random_low = versus_random["paired_bootstrap"]["ci_low"]
    precision_low = precision_noninferiority["paired_bootstrap"]["ci_low"]
    proceed = (
        baseline_low > 0.0
        and random_low > 0.0
        and precision_low > -MAX_CONFIRMATION_PRECISION_DROP
    )
    first_run = next(iter(next(iter(runs.values())).values()))["result"]
    return {
        "schema_version": 1,
        "experiment": "weighted_strategy2_independent_confirmation",
        "domain": first_run["domain"],
        "evaluation_split": "test",
        "locked_weight": locked_weight,
        "maximum_confirmation_precision_drop": MAX_CONFIRMATION_PRECISION_DROP,
        "precision_interval": "one_sided_95_percent_lower_bound",
        "configurations": summarize_configurations(runs),
        "comparisons": {
            "answer_reach_vs_strategy1": versus_baseline,
            "answer_reach_vs_random_weighted": versus_random,
            "selected_pr_auc_noninferiority_vs_strategy1": precision_noninferiority,
        },
        "gate": {
            "answer_reach_vs_strategy1_ci_low": baseline_low,
            "answer_reach_vs_random_weighted_ci_low": random_low,
            "selected_pr_auc_one_sided_ci_low": precision_low,
            "proceed_to_gate_d": proceed,
            "decision": "继续门槛D" if proceed else "不扩展完整训练",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--locked-weight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = confirm_weight(
        args.run_root,
        locked_weight=_read_locked_weight(args.locked_weight),
        protocol_path=args.protocol,
    )
    write_json(report, args.output_dir / "confirmation_summary.json")
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
