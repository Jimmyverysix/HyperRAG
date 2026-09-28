"""Generate LaTeX macros and tables from formal Retriever-only aggregates."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence


METHODS = ("baseline", "matched_random", "ours")
METHOD_LABELS = {
    "baseline": "策略 1",
    "matched_random": "策略 2",
    "ours": "策略 3",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _decimal(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _percent(value: float, digits: int = 2) -> str:
    return _decimal(100.0 * value, digits)


def _count(value: int) -> str:
    return f"{value:,}"


def _macro(name: str, value: object) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _bold_best(values: Sequence[float]) -> list[str]:
    best = max(values)
    return [
        rf"\textbf{{{_percent(value)}}}"
        if abs(value - best) <= 1e-12
        else _percent(value)
        for value in values
    ]


def _preflight_counts(preflight: Mapping[str, Any]) -> tuple[int, int]:
    aligned = 0
    eligible = 0
    for domain in preflight["domains"]:
        for split in domain["splits"].values():
            aligned += int(split["aligned_queries"])
            eligible += int(split["retriever_eligible_queries"])
    return aligned, eligible


def _selection_counts(selection: Mapping[str, Any]) -> tuple[int, int, int]:
    values = [float(row["lambda"]) for row in selection["domains"].values()]
    masking = sum(value == 0.0 for value in values)
    baseline = sum(value == 1.0 for value in values)
    return masking, len(values) - masking - baseline, baseline


def _macro_lines(
    preflight: Mapping[str, Any],
    selection: Mapping[str, Any],
    prevalence: Mapping[str, Any],
    main: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
) -> list[str]:
    aligned, eligible = _preflight_counts(preflight)
    masking, soft, baseline = _selection_counts(selection)
    macro = main["equal_domain_macro"]
    comparisons = main["comparisons"]
    baseline_difference = comparisons["ours_minus_baseline"]
    random_difference = comparisons["ours_minus_matched_random"]

    sensitivity_rows = sensitivity["domains"]
    sensitivity_summary = {}
    for method in ("baseline", "ours"):
        rows = [row for row in sensitivity_rows if row["method"] == method]
        for metric in ("answer_path_mrr", "answer_reach_10"):
            for statistic in ("standard_deviation", "range"):
                sensitivity_summary[(method, metric, statistic)] = fmean(
                    float(row[metric][statistic]) for row in rows
                )

    lines = ["% 自动生成；禁止手工修改实验数字。"]
    values = {
        "DatasetCount": len(preflight["domains"]),
        "AlignedQueryCount": _count(aligned),
        "EligibleQueryCount": _count(eligible),
        "SampledNegativeCount": _count(
            int(prevalence["overall"]["sampled_negative_count"])
        ),
        "PathConsistentNegativeCount": _count(
            int(prevalence["overall"]["path_consistent_negative_count"])
        ),
        "ConflictRate": _percent(
            float(prevalence["overall"]["path_consistent_negative_rate"])
        ),
        "AffectedQueryRate": _percent(
            float(prevalence["overall"]["equal_domain_affected_query_rate"])
        ),
        "MaskSelectedDomainCount": masking,
        "SoftSelectedDomainCount": soft,
        "BaselineSelectedDomainCount": baseline,
    }
    for method, prefix in (
        ("baseline", "StrategyOne"),
        ("matched_random", "StrategyTwo"),
        ("ours", "StrategyThree"),
    ):
        values[f"{prefix}MRR"] = _percent(float(macro[method]["reciprocal_rank"]))
        values[f"{prefix}ReachTen"] = _percent(
            float(macro[method]["answer_reach_10"])
        )
    for comparison, prefix in (
        (baseline_difference, "OursBaseline"),
        (random_difference, "OursRandom"),
    ):
        for metric, suffix in (
            ("reciprocal_rank", "MRR"),
            ("answer_reach_10", "ReachTen"),
        ):
            result = comparison[metric]
            values[f"{prefix}{suffix}Delta"] = _decimal(
                float(result["difference_percentage_points"])
            )
            values[f"{prefix}{suffix}CILow"] = _decimal(
                float(result["ci_low_percentage_points"])
            )
            values[f"{prefix}{suffix}CIHigh"] = _decimal(
                float(result["ci_high_percentage_points"])
            )
    values["StrategyOneSensitivityMRRStd"] = _percent(
        sensitivity_summary[("baseline", "answer_path_mrr", "standard_deviation")]
    )
    values["StrategyThreeSensitivityMRRStd"] = _percent(
        sensitivity_summary[("ours", "answer_path_mrr", "standard_deviation")]
    )
    values["StrategyOneSensitivityMRRRange"] = _percent(
        sensitivity_summary[("baseline", "answer_path_mrr", "range")]
    )
    values["StrategyThreeSensitivityMRRRange"] = _percent(
        sensitivity_summary[("ours", "answer_path_mrr", "range")]
    )
    lines.extend(_macro(name, value) for name, value in values.items())
    return lines


def _main_table(
    selection: Mapping[str, Any],
    main: Mapping[str, Any],
) -> list[str]:
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"领域 & $\lambda_D^*$ & \multicolumn{3}{c}{Answer-Path MRR (\%)} & "
        r"\multicolumn{3}{c}{Reach@10 (\%)} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r" & & 策略 1 & 策略 2 & 策略 3 & 策略 1 & 策略 2 & 策略 3 \\",
        r"\midrule",
    ]
    for row in main["domains"]:
        mrr_values = [float(row[method]["reciprocal_rank"]) for method in METHODS]
        reach_values = [float(row[method]["answer_reach_10"]) for method in METHODS]
        fields = [
            _escape(row["domain"]),
            _decimal(float(selection["domains"][row["domain"]]["lambda"]), 2),
            *_bold_best(mrr_values),
            *_bold_best(reach_values),
        ]
        lines.append(" & ".join(fields) + r" \\")
    macro = main["equal_domain_macro"]
    lines.append(r"\midrule")
    macro_mrr = [float(macro[method]["reciprocal_rank"]) for method in METHODS]
    macro_reach = [float(macro[method]["answer_reach_10"]) for method in METHODS]
    lines.append(
        " & ".join(
            ["等领域宏平均", "--", *_bold_best(macro_mrr), *_bold_best(macro_reach)]
        )
        + r" \\"
    )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return lines


def _prevalence_table(prevalence: Mapping[str, Any]) -> list[str]:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"领域 & 采样负例 & 路径一致负例 & 比例 (\%) & 受影响问题 (\%) \\",
        r"\midrule",
    ]
    for row in prevalence["domains"]:
        lines.append(
            " & ".join(
                [
                    _escape(row["domain"]),
                    _count(int(row["sampled_negative_count"])),
                    _count(int(row["path_consistent_negative_count"])),
                    _percent(float(row["path_consistent_negative_rate"])),
                    _percent(float(row["affected_query_rate_across_seeds"])),
                ]
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return lines


def _lambda_table(selection: Mapping[str, Any]) -> list[str]:
    grid = [float(value) for value in selection["lambda_grid"]]
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        "领域 & $\\lambda_D^*$ & "
        + " & ".join(rf"${value:.2f}$" for value in grid)
        + r" \\",
        r"\midrule",
    ]
    for domain, row in selection["domains"].items():
        selected = float(row["lambda"])
        scores = []
        for value in grid:
            score = float(row["validation"][str(value)]["mean_answer_path_mrr"])
            rendered = _percent(score)
            scores.append(rf"\textbf{{{rendered}}}" if value == selected else rendered)
        lines.append(
            " & ".join([_escape(domain), _decimal(selected), *scores]) + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return lines


def _sensitivity_table(sensitivity: Mapping[str, Any]) -> list[str]:
    by_domain = {}
    for row in sensitivity["domains"]:
        by_domain.setdefault(row["domain"], {})[row["method"]] = row
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"领域 & 问题数 & 策略 1 MRR & 策略 3 MRR & 策略 1 范围 & 策略 3 范围 \\",
        r"\midrule",
    ]
    for domain, methods in by_domain.items():
        baseline = methods["baseline"]
        ours = methods["ours"]
        lines.append(
            " & ".join(
                [
                    _escape(domain),
                    _count(int(baseline["query_count"])),
                    _percent(float(baseline["answer_path_mrr"]["mean"])),
                    _percent(float(ours["answer_path_mrr"]["mean"])),
                    _percent(float(baseline["answer_path_mrr"]["range"])),
                    _percent(float(ours["answer_path_mrr"]["range"])),
                ]
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return lines


def generate_paper_artifacts(
    preflight_path: Path,
    selection_path: Path,
    prevalence_path: Path,
    main_path: Path,
    sensitivity_path: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Write every numeric LaTeX artifact consumed by the paper."""

    preflight = _load(preflight_path)
    selection = _load(selection_path)
    prevalence = _load(prevalence_path)
    main = _load(main_path)
    sensitivity = _load(sensitivity_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {
        "results": output_dir / "results.tex",
        "main": tables / "main_retriever.tex",
        "prevalence": tables / "prevalence.tex",
        "lambda": tables / "lambda_validation.tex",
        "sensitivity": tables / "path_sensitivity.tex",
    }
    contents = {
        "results": _macro_lines(preflight, selection, prevalence, main, sensitivity),
        "main": _main_table(selection, main),
        "prevalence": _prevalence_table(prevalence),
        "lambda": _lambda_table(selection),
        "sensitivity": _sensitivity_table(sensitivity),
    }
    for name, path in outputs.items():
        path.write_text("\n".join(contents[name]) + "\n", encoding="utf-8")
    return {name: str(path) for name, path in outputs.items()}
