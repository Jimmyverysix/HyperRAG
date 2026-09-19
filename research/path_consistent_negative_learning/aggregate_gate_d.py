"""Aggregate the frozen weighted strategy-2 experiment across all domains."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .run_weighted_suite import weight_token
from .statistics import paired_bootstrap_mean_difference
from .weighted_protocol import DEFAULT_PROTOCOL_PATH, load_protocol, phase_protocol
from .weighted_results import (
    compare_configurations,
    load_weighted_runs,
    summarize_configurations,
    write_json,
)


def _read_locked_weight(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle).get("selected_weight")
    if value is None:
        raise ValueError("没有可用于门槛D的锁定权重")
    return float(value)


def _macro_comparison(
    domain_reports: Mapping[str, Mapping[str, Any]],
    *,
    reference: str,
    comparison: str,
    metric: str,
    excluded_domains: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    reference_values = {
        domain: report["configurations"][reference]["mean"][metric]
        for domain, report in domain_reports.items()
        if domain not in excluded_domains
    }
    comparison_values = {
        domain: report["configurations"][comparison]["mean"][metric]
        for domain, report in domain_reports.items()
        if domain not in excluded_domains
    }
    bootstrap = paired_bootstrap_mean_difference(
        reference_values,
        comparison_values,
        n_resamples=10_000,
        seed=20260919,
    )
    return {
        "reference": reference,
        "comparison": comparison,
        "metric": metric,
        "unit": "domain_macro_average",
        "excluded_domains": sorted(excluded_domains),
        "paired_bootstrap": asdict(bootstrap),
    }


def aggregate_gate_d(
    run_root: Path,
    *,
    locked_weight: float,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    settings = phase_protocol(protocol, "gate_d")
    candidate_weights = [
        float(value) for value in phase_protocol(protocol, "selection")["candidate_weights"]
    ]
    if locked_weight not in candidate_weights:
        raise ValueError("门槛D权重不属于冻结的候选网格")
    strategy2 = f"strategy2_weight_{weight_token(locked_weight)}"
    random_control = f"random_weight_{weight_token(locked_weight)}"
    expected_configurations = {"strategy1", strategy2, random_control}
    expected_domains = set(settings["domains"])
    actual_domains = {
        path.name for path in run_root.iterdir() if (path / "runs").is_dir()
    }
    if actual_domains != expected_domains:
        raise ValueError(
            f"门槛D领域集合不一致：{sorted(actual_domains)} != "
            f"{sorted(expected_domains)}"
        )

    domain_reports = {}
    expected_seeds = list(settings["sampler_and_training_seeds"])
    for domain in sorted(expected_domains):
        runs = load_weighted_runs(
            run_root / domain / "runs",
            expected_evaluation_split="test",
        )
        if set(runs) != expected_configurations:
            raise ValueError(f"领域 {domain} 的实验臂不符合冻结协议")
        for configuration, seed_runs in runs.items():
            if sorted(seed_runs) != expected_seeds:
                raise ValueError(f"领域 {domain} 配置 {configuration} 的种子不一致")
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
                    domain,
                    settings["split_seed"],
                    settings["split_scheme"],
                    settings["training"],
                    "weighted_strategy2_gate_d",
                )
                if actual != expected:
                    raise ValueError(f"门槛D结果不符合冻结协议：{run['path']}")
        domain_reports[domain] = {
            "configurations": summarize_configurations(runs),
            "comparisons": {
                "answer_reach_vs_strategy1": compare_configurations(
                    runs,
                    reference="strategy1",
                    comparison=strategy2,
                    metric="answer_reach_10",
                ),
                "answer_reach_vs_random_weighted": compare_configurations(
                    runs,
                    reference=random_control,
                    comparison=strategy2,
                    metric="answer_reach_10",
                ),
                "selected_pr_auc_vs_strategy1": compare_configurations(
                    runs,
                    reference="strategy1",
                    comparison=strategy2,
                    metric="selected_pr_auc",
                ),
            },
        }

    macro = {
        "answer_reach_vs_strategy1": _macro_comparison(
            domain_reports,
            reference="strategy1",
            comparison=strategy2,
            metric="answer_reach_10",
        ),
        "answer_reach_vs_random_weighted": _macro_comparison(
            domain_reports,
            reference=random_control,
            comparison=strategy2,
            metric="answer_reach_10",
        ),
        "selected_pr_auc_vs_strategy1": _macro_comparison(
            domain_reports,
            reference="strategy1",
            comparison=strategy2,
            metric="selected_pr_auc",
        ),
    }
    macro_excluding_confirmation = {
        key: _macro_comparison(
            domain_reports,
            reference=value["reference"],
            comparison=value["comparison"],
            metric=value["metric"],
            excluded_domains=frozenset({"award"}),
        )
        for key, value in macro.items()
    }
    positive_domains = sum(
        report["comparisons"]["answer_reach_vs_strategy1"]["seed_summary"][
            "mean_difference"
        ]
        > 0.0
        for report in domain_reports.values()
    )
    return {
        "schema_version": 1,
        "experiment": "weighted_strategy2_gate_d",
        "locked_weight": locked_weight,
        "domains": domain_reports,
        "macro_comparisons": macro,
        "macro_comparisons_excluding_confirmation_domain": (
            macro_excluding_confirmation
        ),
        "direction_summary": {
            "positive_answer_reach_domains_vs_strategy1": positive_domains,
            "domain_count": len(domain_reports),
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
    report = aggregate_gate_d(
        args.run_root,
        locked_weight=_read_locked_weight(args.locked_weight),
        protocol_path=args.protocol,
    )
    write_json(report, args.output_dir / "gate_d_summary.json")
    print(json.dumps(report["direction_summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
