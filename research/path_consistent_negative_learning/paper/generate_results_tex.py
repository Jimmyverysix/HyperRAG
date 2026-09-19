"""把审计与训练汇总转换为论文可直接引用的 LaTeX 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DOMAIN_LABELS = {
    "art": "艺术",
    "award": "奖项",
    "edu": "教育",
    "health": "健康",
    "infra": "基础设施",
    "loc": "地理",
    "org": "组织",
    "people": "人物",
    "sci": "科学",
    "sport": "体育",
    "tax": "生物分类",
}
STRATEGY_LABELS = {
    "strategy1_negative": "策略1",
    "strategy2_ignore": "策略2",
    "random_drop": "随机丢弃",
    "strategy3_positive": "策略3",
}


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _count(value: int | float) -> str:
    return f"{int(value):,}"


def _percent(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}\\%"


def _mean_std(mean: float, std: float) -> str:
    return f"{100 * mean:.2f} $\\pm$ {100 * std:.2f}"


def _comparison(
    report: Mapping[str, Any], reference: str, metric: str
) -> Mapping[str, Any]:
    return next(
        row
        for row in report["comparisons"]
        if row["reference"] == reference
        and row["comparison"] == "strategy2_ignore"
        and row["metric"] == metric
    )


def render_results(
    audit_report: Mapping[str, Any], training_report: Mapping[str, Any]
) -> str:
    overall = audit_report["overall"]
    gate_b = audit_report["gate_b"]
    gate_c = training_report["gate_c"]
    lines = [
        "% 本文件由 generate_results_tex.py 生成，请勿手工修改。",
        rf"\newcommand{{\AuditDomainCount}}{{{_count(overall['domain_count'])}}}",
        rf"\newcommand{{\AuditQueryCount}}{{{_count(overall['query_count'])}}}",
        rf"\newcommand{{\CandidateCount}}{{{_count(overall['candidate_count'])}}}",
        rf"\newcommand{{\DisputedCandidateCount}}{{{_count(overall['disputed_candidate_count'])}}}",
        rf"\newcommand{{\CandidateRate}}{{{_percent(overall['candidate_rate'], 3)}}}",
        rf"\newcommand{{\SampledCount}}{{{_count(overall['sampled_count'])}}}",
        rf"\newcommand{{\DisputedSampledCount}}{{{_count(overall['disputed_sampled_count'])}}}",
        rf"\newcommand{{\SampledRate}}{{{_percent(overall['sampled_rate'], 2)}}}",
        rf"\newcommand{{\GateBThreshold}}{{{_percent(gate_b['threshold'], 0)}}}",
        rf"\newcommand{{\GateBDecision}}{{{gate_b['decision']}}}",
        "\\newcommand{\\DomainAuditRows}{%",
    ]
    for row in sorted(
        audit_report["domains"], key=lambda item: item["sampled_rate"], reverse=True
    ):
        label = DOMAIN_LABELS.get(row["domain"], row["domain"])
        lines.append(
            f"{label} & {_count(row['query_count'])} & "
            f"{_percent(row['candidate_rate'])} & "
            f"{_percent(row['sampled_seed_mean'])} $\\pm$ "
            f"{100 * row['sampled_seed_std']:.2f} \\\\"  # percentage points
        )
    lines.append("}")

    lines.append("\\newcommand{\\GateCMetricRows}{%")
    for strategy in (
        "strategy1_negative",
        "strategy2_ignore",
        "random_drop",
        "strategy3_positive",
    ):
        values = training_report["strategies"][strategy]
        lines.append(
            f"{STRATEGY_LABELS[strategy]} & "
            f"{_mean_std(values['mean']['answer_reach_10'], values['std']['answer_reach_10'])} & "
            f"{_mean_std(values['mean']['all_shortest_recall_10'], values['std']['all_shortest_recall_10'])} & "
            f"{_mean_std(values['mean']['selected_pr_auc'], values['std']['selected_pr_auc'])} \\\\"
        )
    lines.append("}")

    for reference, macro_prefix in (
        ("strategy1_negative", "VsStrategyOne"),
        ("random_drop", "VsRandomDrop"),
    ):
        result = _comparison(training_report, reference, "answer_reach_10")
        bootstrap = result["paired_bootstrap"]
        lines.extend(
            [
                rf"\newcommand{{\{macro_prefix}Pairs}}{{{_count(bootstrap['n_pairs'])}}}",
                rf"\newcommand{{\{macro_prefix}Difference}}{{{100 * bootstrap['mean_difference']:.2f}}}",
                rf"\newcommand{{\{macro_prefix}CILow}}{{{100 * bootstrap['ci_low']:.2f}}}",
                rf"\newcommand{{\{macro_prefix}CIHigh}}{{{100 * bootstrap['ci_high']:.2f}}}",
            ]
        )
    precision = _comparison(
        training_report, "strategy1_negative", "selected_pr_auc"
    )["seed_summary"]["mean_difference"]
    lines.append(
        rf"\newcommand{{\StrategyTwoPrecisionDifference}}{{{100 * precision:.2f}}}"
    )
    lines.append(rf"\newcommand{{\GateCDecision}}{{{gate_c['decision']}}}")
    lines.append("\\newif\\ifgatecpassed")
    lines.append(
        "\\gatecpassedtrue"
        if gate_c["proceed_to_gate_d"]
        else "\\gatecpassedfalse"
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    content = render_results(_read_json(args.audit), _read_json(args.training))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8", newline="\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
