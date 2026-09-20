"""从经验证的聚合产物生成 WWW 稿件宏与表格，禁止手工抄录实验数字。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _percent(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}"


def _tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _comparison(report: dict[str, Any], key: str) -> dict[str, float]:
    try:
        return report["macro_comparisons"][key]
    except KeyError as exc:
        raise ValueError(f"主结果缺少预注册比较：{key}") from exc


def generate(
    historical_path: Path,
    audit_path: Path,
    proxy_summary_path: Path,
    proxy_table_path: Path,
    selection_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    historical = _load_json(historical_path)
    audit = _load_json(audit_path)
    proxy = _load_json(proxy_summary_path)
    if historical.get("status") != "passed":
        raise ValueError("历史实验校验未通过，拒绝生成论文数字")
    if not audit.get("historical_totals_verified"):
        raise ValueError("逐题结构审计尚未与历史总量核对")
    if proxy.get("scope") != "structured_proxy_only_not_official_hyperrag_qa":
        raise ValueError("主结果 scope 不是冻结的结构代理实验")

    with proxy_table_path.open("r", encoding="utf-8", newline="") as handle:
        proxy_rows = list(csv.DictReader(handle))
    with selection_path.open("r", encoding="utf-8", newline="") as handle:
        selection_rows = list(csv.DictReader(handle))
    domains = list(proxy["domains"])
    if [row["dataset"] for row in proxy_rows] != domains:
        raise ValueError("proxy CSV 的数据集顺序或集合与 summary 不一致")
    if {row["dataset"] for row in selection_rows} != set(domains):
        raise ValueError("lambda selection 与主结果的数据集集合不一致")

    tuned_reach = _comparison(
        proxy,
        "dataset_specific_lambda_star_vs_baseline_lambda_1_answer_reach_10",
    )
    tuned_pr = _comparison(
        proxy,
        "dataset_specific_lambda_star_vs_baseline_lambda_1_selected_pr_auc",
    )
    random_reach = _comparison(
        proxy,
        "dataset_specific_lambda_star_vs_matched_random_at_lambda_star_answer_reach_10",
    )
    facts = historical["facts"]
    overall = audit["overall"]
    correlation = proxy["lambda_conflict_correlation"]
    macros = [
        "% 本文件由 scripts/generate_paper_results.py 生成，请勿手工修改。",
        _macro("DatasetCount", str(len(domains))),
        _macro("AuditQueryCount", f"{int(overall['query_count']):,}"),
        _macro("AuditSampledConflictRate", _percent(overall["sampled_conflict_rate"])),
        _macro(
            "AuditConflictQuestionRate",
            _percent(overall["queries_with_sampled_conflict_any_seed_rate"]),
        ),
        _macro("HistoricalLambdaZeroPRDelta", _percent(facts["lambda_zero_selected_pr_auc_delta"])),
        _macro("HistoricalTransferLambda", f"{facts['historical_transfer_lambda']:g}"),
        _macro("HistoricalTransferReachDelta", _percent(facts["historical_transfer_answer_reach_delta"])),
        _macro("HistoricalTransferPRDelta", _percent(facts["historical_transfer_selected_pr_auc_delta"])),
        _macro("TunedReachDelta", _percent(tuned_reach["mean_difference"])),
        _macro("TunedReachCILow", _percent(tuned_reach["ci_low"])),
        _macro("TunedReachCIHigh", _percent(tuned_reach["ci_high"])),
        _macro("TunedPRDelta", _percent(tuned_pr["mean_difference"])),
        _macro("TunedPRCILow", _percent(tuned_pr["ci_low"])),
        _macro("TunedPRCIHigh", _percent(tuned_pr["ci_high"])),
        _macro("TunedVsRandomReachDelta", _percent(random_reach["mean_difference"])),
        _macro("TunedVsRandomReachCILow", _percent(random_reach["ci_low"])),
        _macro("TunedVsRandomReachCIHigh", _percent(random_reach["ci_high"])),
        _macro("LambdaConflictRho", f"{correlation['spearman_rho']:.2f}"),
        _macro("LambdaConflictP", f"{correlation['p_value']:.3f}"),
    ]

    main_table = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"领域 & $\lambda_D^*$ & 基线 & 屏蔽 & 加权 & 随机 & 全部标正 & $\Delta$PR-AUC \\",
        r"\midrule",
    ]
    for row in proxy_rows:
        main_table.append(
            "{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:+.2f} \\\\".format(
                _tex_escape(row["dataset"]),
                float(row["selected_lambda"]),
                100 * float(row["baseline_answer_reach_10_mean"]),
                100 * float(row["mask_answer_reach_10_mean"]),
                100 * float(row["tuned_answer_reach_10_mean"]),
                100 * float(row["random_answer_reach_10_mean"]),
                100 * float(row["all_positive_answer_reach_10_mean"]),
                100 * float(row["tuned_minus_baseline_selected_pr_auc"]),
            )
        )
    main_table.extend([r"\bottomrule", r"\end{tabular}"])

    audit_table = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"领域 & 问题数 & 采样冲突率（\%） & 涉及冲突的问题（\%） \\",
        r"\midrule",
    ]
    for row in audit["domains"]:
        audit_table.append(
            "{} & {:,} & {:.2f} & {:.2f} \\\\".format(
                _tex_escape(row["domain"]),
                int(row["query_count"]),
                100 * float(row["sampled_conflict_rate"]),
                100 * float(row["queries_with_sampled_conflict_any_seed_rate"]),
            )
        )
    audit_table.extend([r"\bottomrule", r"\end{tabular}"])

    lambda_columns = ["lambda_0", "lambda_0.1", "lambda_0.25", "lambda_0.5", "lambda_0.75", "lambda_1"]
    lambda_table = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"领域 & $\lambda_D^*$ & 0 & .10 & .25 & .50 & .75 & 1.0 \\",
        r"\midrule",
    ]
    for row in selection_rows:
        lambda_table.append(
            "{} & {:.2f} & {} \\\\".format(
                _tex_escape(row["dataset"]),
                float(row["best_lambda"]),
                " & ".join(f"{100 * float(row[column]):.2f}" for column in lambda_columns),
            )
        )
    lambda_table.extend([r"\bottomrule", r"\end{tabular}"])

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    (tables_dir / "main_proxy.tex").write_text("\n".join(main_table) + "\n", encoding="utf-8")
    (tables_dir / "audit.tex").write_text("\n".join(audit_table) + "\n", encoding="utf-8")
    (tables_dir / "lambda.tex").write_text("\n".join(lambda_table) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "sources": {
            "historical": str(historical_path),
            "audit": str(audit_path),
            "proxy_summary": str(proxy_summary_path),
            "proxy_table": str(proxy_table_path),
            "selection": str(selection_path),
        },
        "outputs": [
            "results.tex",
            "tables/main_proxy.tex",
            "tables/audit.tex",
            "tables/lambda.tex",
        ],
        "dataset_count": len(domains),
        "scope": proxy["scope"],
    }
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--proxy-summary", type=Path, required=True)
    parser.add_argument("--proxy-table", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = generate(
        args.historical,
        args.audit,
        args.proxy_summary,
        args.proxy_table,
        args.selection,
        args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
