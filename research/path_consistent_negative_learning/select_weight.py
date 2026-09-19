"""Select a strategy-2 loss weight using only the dedicated selection split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .run_weighted_suite import weight_token
from .weighted_results import (
    QUERY_METRICS,
    compare_configurations,
    compare_query_rows_exact,
    load_weighted_runs,
    summarize_configurations,
    write_json,
)
from .weighted_protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
    phase_protocol,
)


MAX_SELECTION_PRECISION_DROP = 0.008
CONSERVATIVE_TIE_MARGIN = 0.002


def choose_weight(evaluations: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the largest weight within the frozen margin of the global best."""

    eligible = [row for row in evaluations if row["eligible"]]
    if not eligible:
        return None
    best_score = max(row["selection_score"] for row in eligible)
    near_best = [
        row
        for row in eligible
        if row["selection_score"] >= best_score - CONSERVATIVE_TIE_MARGIN
    ]
    return max(near_best, key=lambda item: item["weight"])


def select_weight(
    run_root: Path,
    *,
    candidate_weights: Sequence[float],
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    settings = phase_protocol(protocol, "selection")
    if list(candidate_weights) != [float(value) for value in settings["candidate_weights"]]:
        raise ValueError("候选权重与冻结协议不一致")
    runs = load_weighted_runs(run_root, expected_evaluation_split="selection")
    strategy2_weights = sorted(
        [float(value) for value in settings["candidate_weights"]]
        + [float(value) for value in settings["diagnostic_endpoint_weights"]]
    )
    expected_configurations = {"strategy1", "strategy2_original"}
    expected_configurations.update(
        f"strategy2_weight_{weight_token(weight)}" for weight in strategy2_weights
    )
    expected_configurations.update(
        f"random_weight_{weight_token(float(weight))}"
        for weight in settings["candidate_weights"]
    )
    if set(runs) != expected_configurations:
        raise ValueError(
            "选参实验配置与冻结协议不一致："
            f"{sorted(runs)} != {sorted(expected_configurations)}"
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
                "weighted_strategy2_selection",
            )
            if actual != expected:
                raise ValueError(f"选参结果不符合冻结协议：{run['path']}")
    evaluations = []
    for weight in candidate_weights:
        strategy2 = f"strategy2_weight_{weight_token(weight)}"
        random_control = f"random_weight_{weight_token(weight)}"
        if strategy2 not in runs or random_control not in runs:
            raise ValueError(f"权重 {weight} 缺少策略2或随机降权对照")
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
        precision = compare_configurations(
            runs,
            reference="strategy1",
            comparison=strategy2,
            metric="selected_pr_auc",
        )
        baseline_low = versus_baseline["paired_bootstrap"]["ci_low"]
        random_low = versus_random["paired_bootstrap"]["ci_low"]
        precision_difference = precision["seed_summary"]["mean_difference"]
        eligible = (
            baseline_low > 0.0
            and random_low > 0.0
            and precision_difference >= -MAX_SELECTION_PRECISION_DROP
        )
        evaluations.append(
            {
                "weight": weight,
                "eligible": eligible,
                "selection_score": min(baseline_low, random_low),
                "answer_reach_vs_strategy1": versus_baseline,
                "answer_reach_vs_random_weighted": versus_random,
                "selected_pr_auc_vs_strategy1": precision,
            }
        )

    endpoint_checks: dict[str, Any] = {}
    for name, reference, comparison in (
        ("weight_0_equals_original_strategy2", "strategy2_original", "strategy2_weight_0"),
        ("weight_1_equals_strategy1", "strategy1", "strategy2_weight_1"),
    ):
        endpoint_checks[name] = compare_query_rows_exact(
            runs,
            reference=reference,
            comparison=comparison,
            metrics=QUERY_METRICS,
        )
    endpoints_passed = all(row["passed"] for row in endpoint_checks.values())

    selected = None
    if endpoints_passed:
        selected = choose_weight(evaluations)

    return {
        "schema_version": 1,
        "experiment": "weighted_strategy2_selection",
        "evaluation_split": "selection",
        "candidate_weights": list(candidate_weights),
        "maximum_selection_precision_drop": MAX_SELECTION_PRECISION_DROP,
        "conservative_tie_margin": CONSERVATIVE_TIE_MARGIN,
        "configurations": summarize_configurations(runs),
        "evaluations": evaluations,
        "endpoint_checks": endpoint_checks,
        "endpoint_checks_passed": endpoints_passed,
        "selected_weight": None if selected is None else selected["weight"],
        "proceed_to_confirmation": selected is not None,
        "decision": "进入独立确认" if selected is not None else "停止加权改进",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument(
        "--candidate-weights",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 0.75],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = select_weight(
        args.run_root,
        candidate_weights=args.candidate_weights,
        protocol_path=args.protocol,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(report, args.output_dir / "selection_summary.json")
    write_json(
        {
            "selected_weight": report["selected_weight"],
            "source": str(args.output_dir / "selection_summary.json"),
            "locked_before_confirmation": True,
        },
        args.output_dir / "locked_weight.json",
    )
    print(report["decision"])
    if report["selected_weight"] is not None:
        print(f"selected_weight={report['selected_weight']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
