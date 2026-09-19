"""Generate the publication figure for cross-domain disputed-negative prevalence."""

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


BLUE = "#0072B2"
ORANGE = "#E69F00"
RED = "#D55E00"
GRID = "#D9D9D9"
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


def _font_family() -> str:
    for family in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return family
    return "DejaVu Sans"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [_font_family(), "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.6,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def _label_bars(ax: plt.Axes, values: np.ndarray) -> None:
    offset = max(float(values.max(initial=0.0)) * 0.015, 0.00015)
    for index, value in enumerate(values):
        percentage = 100 * value
        digits = 3 if percentage < 0.01 else 2
        ax.text(
            value + offset,
            index,
            f"{percentage:.{digits}f}%",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#333333",
        )


def generate_figure(report_path: Path, output_dir: Path) -> tuple[Path, Path]:
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    domains = sorted(report["domains"], key=lambda row: row["sampled_rate"])
    names = [
        f"{DOMAIN_LABELS.get(row['domain'], row['domain'])}（{row['domain']}）"
        for row in domains
    ]
    candidate_rates = np.asarray([row["candidate_rate"] for row in domains])
    sampled_rates = np.asarray([row["sampled_seed_mean"] for row in domains])
    sampled_std = np.asarray([row["sampled_seed_std"] for row in domains])
    threshold = float(report["gate_b"]["threshold"])
    y = np.arange(len(names))

    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.3), sharey=True)
    axes[0].barh(y, candidate_rates, color=BLUE, height=0.62)
    axes[1].barh(
        y,
        sampled_rates,
        xerr=sampled_std,
        color=ORANGE,
        height=0.62,
        error_kw={"ecolor": "#4D4D4D", "elinewidth": 0.8, "capsize": 2},
    )
    axes[0].set_yticks(y, labels=names)
    axes[0].set_title("（a）完整候选池")
    axes[1].set_title("（b）实际采样负例（5 个种子）")
    axes[0].set_xlabel("争议候选比例")
    axes[1].set_xlabel("争议负例比例")
    for ax in axes:
        ax.axvline(
            threshold,
            color=RED,
            linestyle="--",
            linewidth=1.2,
            label="门槛 B：1%",
        )
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    _label_bars(axes[0], candidate_rates)
    _label_bars(axes[1], sampled_rates)
    x_max = max(
        float(candidate_rates.max(initial=threshold)),
        float((sampled_rates + sampled_std).max(initial=threshold)),
        threshold,
    )
    for ax in axes:
        ax.set_xlim(0.0, x_max * 1.22 + 0.001)
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("WikiTopics 结构重建中的争议负例发生率", fontsize=11, fontweight="bold")
    fig.subplots_adjust(wspace=0.16, top=0.88)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig_prevalence.pdf"
    png_path = output_dir / "fig_prevalence.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in generate_figure(args.report, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
