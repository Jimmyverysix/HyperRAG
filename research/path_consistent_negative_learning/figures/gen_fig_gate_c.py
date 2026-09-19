"""Generate the publication figure for the four-arm Gate C comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ORDER = (
    "strategy1_negative",
    "strategy2_ignore",
    "random_drop",
    "strategy3_positive",
)
LABELS = ("策略1", "策略2", "随机丢弃", "策略3")
COLORS = ("#A7A9AC", "#D55E00", "#56B4E9", "#009E73")
PANELS = (
    ("answer_reach_10", "（a）答案可达率@10"),
    ("all_shortest_recall_10", "（b）全部最短路径 Recall@10"),
    ("selected_pr_auc", "（c）选中路径 PR-AUC"),
)


def _font_family() -> str:
    for family in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return family
    return "DejaVu Sans"


def generate_figure(summary_path: Path, output_dir: Path) -> tuple[Path, Path]:
    with summary_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [_font_family(), "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    x = np.arange(len(ORDER))
    for ax, (metric, title) in zip(axes, PANELS, strict=True):
        means = np.asarray([report["strategies"][name]["mean"][metric] for name in ORDER])
        stds = np.asarray([report["strategies"][name]["std"][metric] for name in ORDER])
        for index, (value, deviation, color) in enumerate(
            zip(means, stds, COLORS, strict=True)
        ):
            ax.errorbar(
                index,
                value,
                yerr=deviation,
                fmt="o",
                markersize=7,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                ecolor="#444444",
                elinewidth=0.9,
                capsize=2.5,
                zorder=3,
            )
        ax.set_title(title)
        ax.set_xticks(x, LABELS, rotation=24, ha="right")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        lower = max(0.0, float((means - stds).min()) - 0.05)
        upper = min(1.0, float((means + stds).max()) + 0.08)
        if upper - lower < 0.18:
            center = (upper + lower) / 2
            lower, upper = max(0.0, center - 0.09), min(1.0, center + 0.09)
        ax.set_ylim(lower, upper)
        for index, value in enumerate(means):
            ax.text(
                index,
                value + 0.025 * (upper - lower),
                f"{100 * value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )
    fig.suptitle("门槛 C：四种监督处理的结构化代理实验", fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(wspace=0.36, top=0.81, bottom=0.23)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig_gate_c.pdf"
    png_path = output_dir / "fig_gate_c.png"
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
