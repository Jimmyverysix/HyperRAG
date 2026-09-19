"""把全部实验汇总转换为 proposal 主文档可直接引用的 LaTeX 宏。"""

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
BASE_STRATEGY_LABELS = {
    "strategy1_negative": "策略1",
    "strategy2_ignore": "策略2",
    "random_drop": "随机丢弃",
    "strategy3_positive": "策略3",
}
WEIGHTED_STRATEGY_LABELS = {
    "strategy1_negative": "策略1",
    "strategy2_weighted": "策略2加权版",
    "random_weighted": "随机降权",
}


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _count(value: int | float) -> str:
    return f"{int(value):,}"


def _number(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _percentage_points(value: float, digits: int = 2) -> str:
    return f"{100 * value:+.{digits}f}"


def _mean_std(mean: float, std: float) -> str:
    return f"{100 * mean:.2f} $\\pm$ {100 * std:.2f}"


def _percent(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}\\%"


def _adaptive_percent(value: float) -> str:
    percentage = 100 * value
    digits = 3 if percentage < 0.01 else 2
    return f"{percentage:.{digits}f}\\%"


def _base_comparison(
    report: Mapping[str, Any], reference: str, metric: str
) -> Mapping[str, Any]:
    return next(
        row
        for row in report["comparisons"]
        if row["reference"] == reference
        and row["comparison"] == "strategy2_ignore"
        and row["metric"] == metric
    )


def render_base_results(
    audit_report: Mapping[str, Any], training_report: Mapping[str, Any]
) -> str:
    """生成结构审计和原始门槛 C 使用的正文宏。"""

    overall = audit_report["overall"]
    gate_b = audit_report["gate_b"]
    gate_c = training_report["gate_c"]
    lines = [
        "% 本文件由 proposal/generate_results_tex.py 生成，请勿手工修改。",
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
        r"\newcommand{\DomainAuditRows}{%",
    ]
    for row in sorted(
        audit_report["domains"], key=lambda item: item["sampled_rate"], reverse=True
    ):
        label = DOMAIN_LABELS.get(row["domain"], row["domain"])
        lines.append(
            f"{label} & {_count(row['query_count'])} & "
            f"{_adaptive_percent(row['candidate_rate'])} & "
            f"{_percent(row['sampled_seed_mean'])} $\\pm$ "
            f"{100 * row['sampled_seed_std']:.2f} \\\\"
        )
    lines.append("}")

    lines.append(r"\newcommand{\GateCMetricRows}{%")
    for strategy in (
        "strategy1_negative",
        "strategy2_ignore",
        "random_drop",
        "strategy3_positive",
    ):
        values = training_report["strategies"][strategy]
        lines.append(
            f"{BASE_STRATEGY_LABELS[strategy]} & "
            f"{_mean_std(values['mean']['answer_reach_10'], values['std']['answer_reach_10'])} & "
            f"{_mean_std(values['mean']['all_shortest_recall_10'], values['std']['all_shortest_recall_10'])} & "
            f"{_mean_std(values['mean']['selected_pr_auc'], values['std']['selected_pr_auc'])} \\\\"
        )
    lines.append("}")

    for reference, macro_prefix in (
        ("strategy1_negative", "VsStrategyOne"),
        ("random_drop", "VsRandomDrop"),
    ):
        result = _base_comparison(training_report, reference, "answer_reach_10")
        bootstrap = result["paired_bootstrap"]
        lines.extend(
            [
                rf"\newcommand{{\{macro_prefix}Pairs}}{{{_count(bootstrap['n_pairs'])}}}",
                rf"\newcommand{{\{macro_prefix}Difference}}{{{100 * bootstrap['mean_difference']:.2f}}}",
                rf"\newcommand{{\{macro_prefix}CILow}}{{{100 * bootstrap['ci_low']:.2f}}}",
                rf"\newcommand{{\{macro_prefix}CIHigh}}{{{100 * bootstrap['ci_high']:.2f}}}",
            ]
        )
    precision = _base_comparison(
        training_report, "strategy1_negative", "selected_pr_auc"
    )["seed_summary"]["mean_difference"]
    lines.append(
        rf"\newcommand{{\StrategyTwoPrecisionDifference}}{{{100 * precision:.2f}}}"
    )
    lines.append(rf"\newcommand{{\GateCDecision}}{{{gate_c['decision']}}}")
    lines.append(r"\newif\ifgatecpassed")
    lines.append(
        r"\gatecpassedtrue" if gate_c["proceed_to_gate_d"] else r"\gatecpassedfalse"
    )
    return "\n".join(lines) + "\n"


def _interval(comparison: Mapping[str, Any]) -> str:
    bootstrap = comparison["paired_bootstrap"]
    return (
        f"{_percentage_points(bootstrap['mean_difference'])} "
        f"[{_percentage_points(bootstrap['ci_low'])}, "
        f"{_percentage_points(bootstrap['ci_high'])}]"
    )


def _latex_text(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _configuration(
    report: Mapping[str, Any],
    *,
    strategy: str,
    weight: float | None = None,
) -> Mapping[str, Any]:
    matches = []
    for values in report["configurations"].values():
        if values["strategy"] != strategy:
            continue
        actual_weight = values.get("disputed_negative_weight")
        if weight is None and actual_weight is None:
            matches.append(values)
        elif weight is not None and actual_weight is not None:
            if abs(float(actual_weight) - weight) <= 1e-12:
                matches.append(values)
    if len(matches) != 1:
        raise ValueError(
            f"策略 {strategy!r}、权重 {weight!r} 应恰好对应一个配置，"
            f"实际找到 {len(matches)} 个"
        )
    return matches[0]


def _render_selection(report: Mapping[str, Any]) -> list[str]:
    selected_weight = report.get("selected_weight")
    lines = [
        rf"\newcommand{{\WeightedSelectionCandidateCount}}{{{_count(len(report['evaluations']))}}}",
        rf"\newcommand{{\WeightedSelectionPrecisionLimit}}{{{100 * float(report['maximum_selection_precision_drop']):.2f}}}",
        rf"\newcommand{{\WeightedSelectionTieMargin}}{{{100 * float(report['conservative_tie_margin']):.2f}}}",
        rf"\newcommand{{\WeightedSelectedWeight}}{{{'--' if selected_weight is None else _number(float(selected_weight))}}}",
        rf"\newcommand{{\WeightedSelectionDecision}}{{{_latex_text(report['decision'])}}}",
        rf"\newcommand{{\WeightedEndpointDecision}}{{{'通过' if report['endpoint_checks_passed'] else '未通过'}}}",
        r"\newif\ifweightedselectionpassed",
        (
            r"\weightedselectionpassedtrue"
            if report["proceed_to_confirmation"]
            else r"\weightedselectionpassedfalse"
        ),
        r"\newcommand{\WeightedSelectionRows}{%",
    ]
    for evaluation in sorted(report["evaluations"], key=lambda row: row["weight"]):
        weight = float(evaluation["weight"])
        weighted = _configuration(
            report, strategy="strategy2_weighted", weight=weight
        )
        random_control = _configuration(
            report, strategy="random_weighted", weight=weight
        )
        lines.append(
            f"{_number(weight)} & "
            f"{_mean_std(weighted['mean']['answer_reach_10'], weighted['std']['answer_reach_10'])} & "
            f"{_mean_std(random_control['mean']['answer_reach_10'], random_control['std']['answer_reach_10'])} & "
            f"{_interval(evaluation['answer_reach_vs_strategy1'])} & "
            f"{_interval(evaluation['answer_reach_vs_random_weighted'])} & "
            f"{_percentage_points(evaluation['selected_pr_auc_vs_strategy1']['seed_summary']['mean_difference'])} & "
            f"{'是' if evaluation['eligible'] else '否'} \\\\"
        )
    lines.append("}")
    return lines


def _render_confirmation(report: Mapping[str, Any] | None) -> list[str]:
    if report is None:
        return [
            r"\newif\ifweightedconfirmationavailable",
            r"\weightedconfirmationavailablefalse",
            r"\newcommand{\WeightedConfirmationDomain}{未运行}",
            r"\newcommand{\WeightedConfirmationWeight}{--}",
            r"\newcommand{\WeightedConfirmationPrecisionLimit}{--}",
            r"\newcommand{\WeightedConfirmationPairs}{0}",
            r"\newcommand{\WeightedConfirmationVsStrategyOne}{未运行}",
            r"\newcommand{\WeightedConfirmationVsRandom}{未运行}",
            r"\newcommand{\WeightedConfirmationPrecision}{未运行}",
            r"\newcommand{\WeightedConfirmationDecision}{未运行}",
            r"\newif\ifweightedconfirmationpassed",
            r"\weightedconfirmationpassedfalse",
            r"\newcommand{\WeightedConfirmationRows}{%",
            r"\multicolumn{3}{c}{未运行} \\ ",
            "}",
        ]

    locked_weight = float(report["locked_weight"])
    comparisons = report["comparisons"]
    versus_strategy1 = comparisons["answer_reach_vs_strategy1"]
    versus_random = comparisons["answer_reach_vs_random_weighted"]
    precision = comparisons["selected_pr_auc_noninferiority_vs_strategy1"]
    lines = [
        r"\newif\ifweightedconfirmationavailable",
        r"\weightedconfirmationavailabletrue",
        rf"\newcommand{{\WeightedConfirmationDomain}}{{{_latex_text(DOMAIN_LABELS.get(report['domain'], report['domain']))}}}",
        rf"\newcommand{{\WeightedConfirmationWeight}}{{{_number(locked_weight)}}}",
        rf"\newcommand{{\WeightedConfirmationPrecisionLimit}}{{{100 * float(report['maximum_confirmation_precision_drop']):.2f}}}",
        rf"\newcommand{{\WeightedConfirmationPairs}}{{{_count(versus_strategy1['paired_bootstrap']['n_pairs'])}}}",
        rf"\newcommand{{\WeightedConfirmationVsStrategyOne}}{{{_interval(versus_strategy1)}}}",
        rf"\newcommand{{\WeightedConfirmationVsRandom}}{{{_interval(versus_random)}}}",
        rf"\newcommand{{\WeightedConfirmationPrecision}}{{{_interval(precision)}}}",
        rf"\newcommand{{\WeightedConfirmationDecision}}{{{_latex_text(report['gate']['decision'])}}}",
        r"\newif\ifweightedconfirmationpassed",
        (
            r"\weightedconfirmationpassedtrue"
            if report["gate"]["proceed_to_gate_d"]
            else r"\weightedconfirmationpassedfalse"
        ),
        r"\newcommand{\WeightedConfirmationRows}{%",
    ]
    configurations = (
        _configuration(report, strategy="strategy1_negative"),
        _configuration(report, strategy="strategy2_weighted", weight=locked_weight),
        _configuration(report, strategy="random_weighted", weight=locked_weight),
    )
    for values in configurations:
        label = WEIGHTED_STRATEGY_LABELS[values["strategy"]]
        lines.append(
            f"{label} & "
            f"{_mean_std(values['mean']['answer_reach_10'], values['std']['answer_reach_10'])} & "
            f"{_mean_std(values['mean']['selected_pr_auc'], values['std']['selected_pr_auc'])} \\\\"
        )
    lines.append("}")
    return lines


def _render_gate_d(report: Mapping[str, Any] | None) -> list[str]:
    lines = [r"\newif\ifgatedavailable"]
    if report is None:
        lines.extend(
            [
                r"\gatedavailablefalse",
                r"\newcommand{\GateDAvailability}{未运行}",
            ]
        )
        return lines

    direction = report["direction_summary"]
    macro = report["macro_comparisons"]
    macro_excluding_confirmation = report[
        "macro_comparisons_excluding_confirmation_domain"
    ]
    lines.extend(
        [
            r"\gatedavailabletrue",
            r"\newcommand{\GateDAvailability}{已完成}",
            rf"\newcommand{{\GateDLockedWeight}}{{{_number(float(report['locked_weight']))}}}",
            rf"\newcommand{{\GateDDomainCount}}{{{_count(direction['domain_count'])}}}",
            rf"\newcommand{{\GateDPositiveDomainCount}}{{{_count(direction['positive_answer_reach_domains_vs_strategy1'])}}}",
            rf"\newcommand{{\GateDVsStrategyOne}}{{{_interval(macro['answer_reach_vs_strategy1'])}}}",
            rf"\newcommand{{\GateDVsRandomWeighted}}{{{_interval(macro['answer_reach_vs_random_weighted'])}}}",
            rf"\newcommand{{\GateDPrecisionVsStrategyOne}}{{{_interval(macro['selected_pr_auc_vs_strategy1'])}}}",
            rf"\newcommand{{\GateDExcludingConfirmationVsStrategyOne}}{{{_interval(macro_excluding_confirmation['answer_reach_vs_strategy1'])}}}",
            rf"\newcommand{{\GateDExcludingConfirmationVsRandomWeighted}}{{{_interval(macro_excluding_confirmation['answer_reach_vs_random_weighted'])}}}",
            rf"\newcommand{{\GateDExcludingConfirmationPrecisionVsStrategyOne}}{{{_interval(macro_excluding_confirmation['selected_pr_auc_vs_strategy1'])}}}",
            r"\newcommand{\GateDDomainRows}{%",
        ]
    )
    for domain, domain_report in sorted(report["domains"].items()):
        locked_weight = float(report["locked_weight"])
        weighted = _configuration(
            domain_report, strategy="strategy2_weighted", weight=locked_weight
        )
        comparisons = domain_report["comparisons"]
        lines.append(
            f"{_latex_text(DOMAIN_LABELS.get(domain, domain))} & "
            f"{_mean_std(weighted['mean']['answer_reach_10'], weighted['std']['answer_reach_10'])} & "
            f"{_percentage_points(comparisons['answer_reach_vs_strategy1']['seed_summary']['mean_difference'])} & "
            f"{_percentage_points(comparisons['answer_reach_vs_random_weighted']['seed_summary']['mean_difference'])} & "
            f"{_percentage_points(comparisons['selected_pr_auc_vs_strategy1']['seed_summary']['mean_difference'])} \\\\"
        )
    lines.extend(["}", r"\newcommand{\GateDMacroRows}{%"])
    macro_labels = (
        ("answer_reach_vs_strategy1", "答案可达率：相对策略1"),
        ("answer_reach_vs_random_weighted", "答案可达率：相对随机降权"),
        ("selected_pr_auc_vs_strategy1", "选中路径曲线下面积：相对策略1"),
    )
    for key, label in macro_labels:
        lines.append(
            f"{label} & {_interval(report['macro_comparisons'][key])} & "
            f"{_interval(report['macro_comparisons_excluding_confirmation_domain'][key])} \\\\"
        )
    lines.append("}")
    return lines


def render_results(
    audit_report: Mapping[str, Any],
    training_report: Mapping[str, Any],
    selection_report: Mapping[str, Any],
    confirmation_report: Mapping[str, Any] | None = None,
    gate_d_report: Mapping[str, Any] | None = None,
) -> str:
    """Render the original results plus weighted-strategy experiment results."""

    selected_weight = selection_report.get("selected_weight")
    if confirmation_report is not None:
        if selected_weight is None:
            raise ValueError("存在独立确认结果时，选参汇总必须包含锁定权重")
        confirmed_weight = float(confirmation_report["locked_weight"])
        if abs(float(selected_weight) - confirmed_weight) > 1e-12:
            raise ValueError("选参和独立确认的锁定权重不一致")
    if gate_d_report is not None:
        if confirmation_report is None:
            raise ValueError("存在门槛D结果时必须同时提供独立确认结果")
        confirmed_weight = float(confirmation_report["locked_weight"])
        gate_d_weight = float(gate_d_report["locked_weight"])
        if abs(confirmed_weight - gate_d_weight) > 1e-12:
            raise ValueError("独立确认和门槛D的锁定权重不一致")

    base = render_base_results(audit_report, training_report).rstrip()
    lines = [
        base,
        "",
        "% 以下宏来自策略2加权改进实验。",
        *_render_selection(selection_report),
        *_render_confirmation(confirmation_report),
        *_render_gate_d(gate_d_report),
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--gate-d", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("generated_results.tex"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gate_d_report = _read_json(args.gate_d) if args.gate_d is not None else None
    confirmation_report = (
        _read_json(args.confirmation) if args.confirmation is not None else None
    )
    content = render_results(
        _read_json(args.audit),
        _read_json(args.training),
        _read_json(args.selection),
        confirmation_report,
        gate_d_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8", newline="\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
