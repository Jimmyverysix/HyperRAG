"""Cross-domain aggregation and Gate B decision for WikiTopics audits."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence


GATE_B_RATE = 0.01


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _sampling_by_seed(summary: Mapping[str, Any]) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for row in summary["sampling_by_seed"]:
        seed = int(row["seed"])
        if seed in result:
            raise ValueError(f"领域 {summary['domain']} 的随机种子 {seed} 重复")
        result[seed] = (
            int(row["sampled_count"]),
            int(row["disputed_sampled_count"]),
        )
    if not result:
        raise ValueError(f"领域 {summary['domain']} 没有采样统计")
    return result


def load_domain_summaries(input_dir: Path) -> tuple[dict[str, Any], ...]:
    summaries = []
    for path in sorted(input_dir.glob("*.summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary.get("domain") != path.name.removesuffix(".summary.json"):
            raise ValueError(f"领域名与文件名不一致：{path}")
        summaries.append(summary)
    if not summaries:
        raise ValueError(f"没有找到领域汇总：{input_dir}")
    return tuple(summaries)


@dataclass(frozen=True)
class DomainAuditRow:
    domain: str
    query_count: int
    candidate_count: int
    disputed_candidate_count: int
    candidate_rate: float
    sampled_count: int
    disputed_sampled_count: int
    sampled_rate: float
    sampled_seed_mean: float
    sampled_seed_std: float

    def to_dict(self) -> dict[str, int | float | str]:
        return dict(self.__dict__)


def _domain_row(summary: Mapping[str, Any]) -> DomainAuditRow:
    by_seed = _sampling_by_seed(summary)
    sampled_count = sum(values[0] for values in by_seed.values())
    disputed_sampled = sum(values[1] for values in by_seed.values())
    seed_rates = [_ratio(disputed, sampled) for sampled, disputed in by_seed.values()]
    candidate_count = int(summary["candidate_pool_size"])
    disputed_candidate = int(summary["disputed_candidate_count"])
    return DomainAuditRow(
        domain=str(summary["domain"]),
        query_count=int(summary["query_count"]),
        candidate_count=candidate_count,
        disputed_candidate_count=disputed_candidate,
        candidate_rate=_ratio(disputed_candidate, candidate_count),
        sampled_count=sampled_count,
        disputed_sampled_count=disputed_sampled,
        sampled_rate=_ratio(disputed_sampled, sampled_count),
        sampled_seed_mean=mean(seed_rates),
        sampled_seed_std=stdev(seed_rates) if len(seed_rates) > 1 else 0.0,
    )


def aggregate_summaries(
    summaries: Sequence[Mapping[str, Any]],
    gate_rate: float = GATE_B_RATE,
) -> dict[str, Any]:
    if not 0.0 <= gate_rate <= 1.0:
        raise ValueError("门槛比例必须位于 [0, 1]")
    rows = tuple(_domain_row(summary) for summary in summaries)
    domains = [row.domain for row in rows]
    if len(domains) != len(set(domains)):
        raise ValueError("领域名重复")

    expected_seeds = set(_sampling_by_seed(summaries[0]))
    seed_totals = {seed: [0, 0] for seed in sorted(expected_seeds)}
    for summary in summaries:
        sampling = _sampling_by_seed(summary)
        if set(sampling) != expected_seeds:
            raise ValueError("各领域必须使用完全相同的随机种子")
        for seed, (sampled, disputed) in sampling.items():
            seed_totals[seed][0] += sampled
            seed_totals[seed][1] += disputed

    candidate_count = sum(row.candidate_count for row in rows)
    disputed_candidate = sum(row.disputed_candidate_count for row in rows)
    sampled_count = sum(row.sampled_count for row in rows)
    disputed_sampled = sum(row.disputed_sampled_count for row in rows)
    affected_domains = sorted(row.domain for row in rows if row.sampled_rate >= gate_rate)
    global_sampled_rate = _ratio(disputed_sampled, sampled_count)
    proceed = global_sampled_rate >= gate_rate or bool(affected_domains)

    return {
        "schema_version": 1,
        "gate_b": {
            "threshold": gate_rate,
            "global_sampled_rate": global_sampled_rate,
            "domains_at_or_above_threshold": affected_domains,
            "proceed_to_causal_training": proceed,
            "decision": "继续门槛C" if proceed else "停止完整重训练并报告负结果",
        },
        "overall": {
            "domain_count": len(rows),
            "query_count": sum(row.query_count for row in rows),
            "candidate_count": candidate_count,
            "disputed_candidate_count": disputed_candidate,
            "candidate_rate": _ratio(disputed_candidate, candidate_count),
            "sampled_count": sampled_count,
            "disputed_sampled_count": disputed_sampled,
            "sampled_rate": global_sampled_rate,
        },
        "sampling_by_seed": [
            {
                "seed": seed,
                "sampled_count": totals[0],
                "disputed_sampled_count": totals[1],
                "disputed_sampled_rate": _ratio(totals[1], totals[0]),
            }
            for seed, totals in seed_totals.items()
        ],
        "domains": [row.to_dict() for row in sorted(rows, key=lambda item: item.domain)],
    }


def write_aggregate(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "combined_summary.json"
    csv_path = output_dir / "combined_domains.csv"
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    domain_rows = list(report["domains"])
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(domain_rows[0]))
        writer.writeheader()
        writer.writerows(domain_rows)


def aggregate_directory(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    report = aggregate_summaries(load_domain_summaries(input_dir))
    write_aggregate(report, output_dir)
    return report

