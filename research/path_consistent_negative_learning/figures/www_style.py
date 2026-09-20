"""WWW 修订版论文图的统一样式与导出。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "baseline": "#4D4D4D",
    "ours": "#0F4D92",
    "ours_soft": "#8FB5D9",
    "random": "#7884B4",
    "mask": "#B64342",
    "positive": "#8BCF8B",
    "gain": "#2E9E44",
    "drop": "#E53935",
    "neutral": "#767676",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Microsoft YaHei",
                "SimHei",
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
            ],
            "font.size": 8,
            "axes.linewidth": 0.9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_label(axis, label: str) -> None:
    axis.text(
        -0.13,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def save_vector_figure(figure, output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_stem.with_suffix(".svg")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text(
        "\n".join(
            line.rstrip()
            for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return svg_path, pdf_path
