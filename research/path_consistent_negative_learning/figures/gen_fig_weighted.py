"""根据选参汇总生成策略2加权改进的出版级权衡图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# Okabe--Ito 色盲友好色板；灰色仅用于阈值和参考线。
COLORS = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#D55E00",
    "#0072B2",
    "#CC79A7",
    "#F0E442",
)
REFERENCE_COLOR = "#666666"


def _font_family() -> str:
    for family in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return family
    return "DejaVu Sans"


def _load_report(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _points(report: Mapping[str, Any]) -> list[dict[str, float | bool]]:
    points = []
    for evaluation in sorted(report["evaluations"], key=lambda row: row["weight"]):
        versus_strategy1 = evaluation["answer_reach_vs_strategy1"]["paired_bootstrap"]
        versus_random = evaluation["answer_reach_vs_random_weighted"]["paired_bootstrap"]
        precision = evaluation["selected_pr_auc_vs_strategy1"]["seed_summary"]
        points.append(
            {
                "weight": float(evaluation["weight"]),
                "eligible": bool(evaluation["eligible"]),
                "precision_difference": 100 * float(precision["mean_difference"]),
                "baseline_difference": 100 * float(versus_strategy1["mean_difference"]),
                "baseline_ci_low": 100 * float(versus_strategy1["ci_low"]),
                "baseline_ci_high": 100 * float(versus_strategy1["ci_high"]),
                "random_difference": 100 * float(versus_random["mean_difference"]),
                "random_ci_low": 100 * float(versus_random["ci_low"]),
                "random_ci_high": 100 * float(versus_random["ci_high"]),
            }
        )
    if not points:
        raise ValueError("选参汇总没有候选权重评估")
    return points


def _asymmetric_error(point: Mapping[str, float | bool], prefix: str) -> np.ndarray:
    center = float(point[f"{prefix}_difference"])
    return np.asarray(
        [
            [center - float(point[f"{prefix}_ci_low"])],
            [float(point[f"{prefix}_ci_high"]) - center],
        ]
    )


def generate_figure(summary_path: Path, output_dir: Path) -> tuple[Path, Path]:
    report = _load_report(summary_path)
    points = _points(report)
    selected_weight = report.get("selected_weight")
    precision_limit = 100 * float(report["maximum_selection_precision_drop"])

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [_font_family(), "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )

    fig, (tradeoff_ax, control_ax) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for index, point in enumerate(points):
        color = COLORS[index % len(COLORS)]
        is_selected = (
            selected_weight is not None
            and abs(float(point["weight"]) - float(selected_weight)) <= 1e-12
        )
        marker = "*" if is_selected else ("o" if point["eligible"] else "X")
        marker_size = 12 if is_selected else 7
        marker_edge = "#222222" if is_selected else "white"
        tradeoff_ax.errorbar(
            float(point["precision_difference"]),
            float(point["baseline_difference"]),
            yerr=_asymmetric_error(point, "baseline"),
            fmt=marker,
            markersize=marker_size,
            color=color,
            markeredgecolor=marker_edge,
            markeredgewidth=0.8,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.5,
            zorder=3,
        )
        tradeoff_ax.annotate(
            rf"$\lambda={float(point['weight']):g}$",
            (
                float(point["precision_difference"]),
                float(point["baseline_difference"]),
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#333333",
        )

        control_ax.errorbar(
            float(point["weight"]),
            float(point["random_difference"]),
            yerr=_asymmetric_error(point, "random"),
            fmt=marker,
            markersize=marker_size,
            color=color,
            markeredgecolor=marker_edge,
            markeredgewidth=0.8,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.5,
            zorder=3,
        )

    tradeoff_ax.axvline(
        -precision_limit,
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )
    tradeoff_ax.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.8, zorder=1)
    tradeoff_ax.set_title("（a）覆盖提升与排序精度的权衡")
    tradeoff_ax.set_xlabel("选中路径 PR-AUC 相对策略1的变化（百分点）")
    tradeoff_ax.set_ylabel("答案可达率@10相对策略1的变化（百分点）")
    tradeoff_ax.text(
        -precision_limit,
        0.98,
        f" 选参保护线：−{precision_limit:.1f}",
        transform=tradeoff_ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=7.5,
        color=REFERENCE_COLOR,
    )
    tradeoff_ax.grid(axis="both")

    control_ax.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.8, zorder=1)
    weights = [float(point["weight"]) for point in points]
    control_ax.set_xticks(weights)
    control_ax.set_xticklabels([f"{weight:g}" for weight in weights])
    control_ax.set_title("（b）相对匹配随机降权的路径信息增益")
    control_ax.set_xlabel(r"争议负例权重 $\lambda$")
    control_ax.set_ylabel("答案可达率@10差值（百分点）")
    control_ax.grid(axis="y")
    control_ax.grid(axis="x", visible=False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#009E73",
            markeredgecolor="white",
            markersize=7,
            label="满足选参条件",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="none",
            markerfacecolor="#D55E00",
            markeredgecolor="white",
            markersize=7,
            label="不满足选参条件",
        ),
    ]
    if selected_weight is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="none",
                markerfacecolor="#E69F00",
                markeredgecolor="#222222",
                markersize=11,
                label=f"锁定权重 {float(selected_weight):g}",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("策略2加权改进：冻结选参集结果", fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(wspace=0.34, top=0.82, bottom=0.27)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig_weighted.pdf"
    png_path = output_dir / "fig_weighted.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in generate_figure(args.summary, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
