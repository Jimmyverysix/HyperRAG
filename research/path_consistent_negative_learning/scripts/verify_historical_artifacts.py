"""交叉验证历史实验产物，并导出机器可读的事实快照。

本脚本只读取 ``artifacts/audit``、``artifacts/gate_c``、
``artifacts/weighted`` 与 ``artifacts/provenance.json``。它不会重写任何历史文件。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


ARTIFACT_FILES = {
    "audit": Path("audit/combined_summary.json"),
    "gate_c": Path("gate_c/training_summary.json"),
    "selection": Path("weighted/selection_summary.json"),
    "confirmation": Path("weighted/confirmation_summary.json"),
    "gate_d": Path("weighted/gate_d_summary.json"),
    "locked_weight": Path("weighted/locked_weight.json"),
    "provenance": Path("provenance.json"),
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"历史产物必须是 JSON object: {path}")
    return payload


def _close(actual: float, expected: float, *, atol: float = 1e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol)


def _comparison(
    comparisons: Iterable[dict[str, Any]],
    comparison: str,
    metric: str,
    *,
    reference: str | None = None,
) -> dict[str, Any]:
    matches = [
        row
        for row in comparisons
        if row.get("comparison") == comparison
        and row.get("metric") == metric
        and (reference is None or row.get("reference") == reference)
    ]
    if len(matches) != 1:
        raise ValueError(
            "无法唯一定位历史比较: "
            f"comparison={comparison}, metric={metric}, reference={reference}"
        )
    return matches[0]


def verify_historical_artifacts(artifacts_dir: Path) -> dict[str, Any]:
    """返回历史事实与逐项一致性检查；任何失败都会抛出 ``ValueError``。"""

    payloads = {
        name: _load_json(artifacts_dir / relative)
        for name, relative in ARTIFACT_FILES.items()
    }
    audit = payloads["audit"]
    gate_c = payloads["gate_c"]
    selection = payloads["selection"]
    confirmation = payloads["confirmation"]
    gate_d = payloads["gate_d"]
    locked = payloads["locked_weight"]
    provenance = payloads["provenance"]

    domain_rows = audit["domains"]
    overall = audit["overall"]
    provenance_domains = provenance["data"]["domains"]
    row_domains = [row["domain"] for row in domain_rows]
    checks: dict[str, bool] = {}

    checks["domain_order_matches_provenance"] = row_domains == provenance_domains
    checks["domain_count_matches_rows"] = (
        overall["domain_count"] == len(domain_rows) == len(provenance_domains)
    )
    for field in (
        "query_count",
        "candidate_count",
        "disputed_candidate_count",
        "sampled_count",
        "disputed_sampled_count",
    ):
        checks[f"audit_{field}_is_domain_sum"] = overall[field] == sum(
            row[field] for row in domain_rows
        )
    checks["audit_candidate_rate_recomputed"] = _close(
        overall["candidate_rate"],
        overall["disputed_candidate_count"] / overall["candidate_count"],
    )
    checks["audit_sampled_rate_recomputed"] = _close(
        overall["sampled_rate"],
        overall["disputed_sampled_count"] / overall["sampled_count"],
    )

    ignore_precision = _comparison(
        gate_c["comparisons"],
        "strategy2_ignore",
        "selected_pr_auc",
        reference="strategy1_negative",
    )["paired_bootstrap"]["mean_difference"]
    gate_c_record = gate_c["gate_c"]
    checks["gate_c_precision_difference_matches_comparison"] = _close(
        ignore_precision,
        gate_c_record["strategy2_vs_strategy1_precision_difference"],
    )
    checks["lambda_zero_failed_registered_gate"] = (
        gate_c_record["proceed_to_gate_d"] is False
        and ignore_precision < -gate_c_record["maximum_allowed_precision_drop"]
    )

    selected_weight = float(selection["selected_weight"])
    provenance_weight = float(
        provenance["weighted_revision"]["selected_disputed_negative_weight"]
    )
    locked_weight = float(locked["selected_weight"])
    checks["historical_weight_consistent"] = (
        _close(selected_weight, provenance_weight)
        and _close(selected_weight, locked_weight)
        and _close(selected_weight, float(confirmation["locked_weight"]))
        and _close(selected_weight, float(gate_d["locked_weight"]))
    )
    checks["historical_selection_used_selection_split"] = (
        selection["evaluation_split"] == "selection"
    )
    checks["historical_confirmation_passed"] = (
        confirmation["evaluation_split"] == "test"
        and confirmation["gate"]["proceed_to_gate_d"] is True
    )

    macro = gate_d["macro_comparisons"]
    reach_delta = macro["answer_reach_vs_strategy1"]["paired_bootstrap"]
    precision_delta = macro["selected_pr_auc_vs_strategy1"]["paired_bootstrap"]
    checks["gate_d_has_all_domains"] = (
        gate_d["direction_summary"]["domain_count"] == len(provenance_domains)
        and len(gate_d["domains"]) == len(provenance_domains)
    )
    checks["gate_d_reports_transfer_not_universal_improvement"] = (
        gate_d["direction_summary"]["positive_answer_reach_domains_vs_strategy1"]
        < len(provenance_domains)
    )

    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("历史产物交叉验证失败: " + ", ".join(failed))

    sampled_rates = [float(row["sampled_rate"]) for row in domain_rows]
    return {
        "schema_version": 1,
        "status": "passed",
        "sources": {
            name: str(relative).replace("\\", "/")
            for name, relative in ARTIFACT_FILES.items()
        },
        "checks": checks,
        "facts": {
            "domain_count": int(overall["domain_count"]),
            "query_count": int(overall["query_count"]),
            "candidate_count": int(overall["candidate_count"]),
            "disputed_candidate_count": int(overall["disputed_candidate_count"]),
            "disputed_candidate_rate": float(overall["candidate_rate"]),
            "sampled_negative_count": int(overall["sampled_count"]),
            "disputed_sampled_negative_count": int(
                overall["disputed_sampled_count"]
            ),
            "disputed_sampled_negative_rate": float(overall["sampled_rate"]),
            "domain_sampled_rate_min": min(sampled_rates),
            "domain_sampled_rate_max": max(sampled_rates),
            "lambda_zero_selected_pr_auc_delta": float(ignore_precision),
            "lambda_zero_gate_c_decision": gate_c_record["decision"],
            "historical_transfer_lambda": selected_weight,
            "historical_transfer_answer_reach_delta": float(
                reach_delta["mean_difference"]
            ),
            "historical_transfer_answer_reach_ci": [
                float(reach_delta["ci_low"]),
                float(reach_delta["ci_high"]),
            ],
            "historical_transfer_selected_pr_auc_delta": float(
                precision_delta["mean_difference"]
            ),
            "historical_transfer_selected_pr_auc_ci": [
                float(precision_delta["ci_low"]),
                float(precision_delta["ci_high"]),
            ],
            "positive_reach_domains": int(
                gate_d["direction_summary"][
                    "positive_answer_reach_domains_vs_strategy1"
                ]
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repository_root = Path(__file__).resolve().parents[3]
    research_root = repository_root / "research" / "path_consistent_negative_learning"
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=research_root / "artifacts",
        help="历史 artifacts 根目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=research_root
        / "artifacts"
        / "www_revision"
        / "provenance"
        / "historical_verification.json",
        help="新命名空间中的验证报告路径",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = verify_historical_artifacts(args.artifacts_dir.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"历史产物验证通过：{len(report['checks'])} 项；报告写入 {output}")


if __name__ == "__main__":
    main()
