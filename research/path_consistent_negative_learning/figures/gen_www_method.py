"""绘制路径一致弱负例重加权的方法示意图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from .www_style import COLORS, apply_style, save_vector_figure


def _entity(axis, x: float, y: float, label: str, color: str) -> None:
    axis.add_patch(Circle((x, y), 0.19, facecolor="white", edgecolor=color, linewidth=1.4))
    axis.text(x, y, label, ha="center", va="center", fontsize=8)


def _fact(axis, x: float, y: float, label: str, color: str) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x - 0.18, y - 0.13),
            0.36,
            0.26,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
        )
    )
    axis.text(x, y, label, ha="center", va="center", fontsize=7)


def _arrow(axis, start: tuple[float, float], end: tuple[float, float], color: str, style: str = "-") -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.3,
            linestyle=style,
            color=color,
            shrinkA=11,
            shrinkB=11,
        )
    )


def generate(output_stem: Path) -> tuple[Path, Path]:
    apply_style()
    figure, axis = plt.subplots(figsize=(7.2, 3.25))
    axis.set_xlim(-0.5, 7.5)
    axis.set_ylim(-0.6, 3.5)
    axis.axis("off")

    rows = (
        (2.8, COLORS["baseline"], "-", "被选中的最短路径", "正例，权重 1", (r"$e_1$", r"$e_3$")),
        (1.6, COLORS["ours"], "--", "未被选中的等长最短路径", "路径一致负例，权重 λ", (r"$e_2$", r"$e_4$")),
        (0.4, COLORS["mask"], ":", "偏离任意最短路径的转移", "普通负例，权重 1", (r"$e_5$", r"$e_6$")),
    )
    for y, color, line_style, title, supervision, facts in rows:
        _entity(axis, 0.4, y, "s", color)
        _fact(axis, 1.8, y, facts[0], color)
        _entity(axis, 3.2, y, "x" if y != 1.6 else "y", color)
        _fact(axis, 4.6, y, facts[1], color)
        _entity(axis, 6.0, y, "a" if y != 0.4 else "z", color)
        for left, right in ((0.55, 1.62), (1.98, 3.01), (3.39, 4.42), (4.78, 5.81)):
            _arrow(axis, (left, y), (right, y), color, line_style)
        axis.text(6.45, y + 0.09, title, color=color, va="center", fontsize=8)
        axis.text(6.45, y - 0.13, supervision, color=COLORS["neutral"], va="center", fontsize=7)

    axis.text(0.4, 3.28, "主题实体", ha="center", color=COLORS["neutral"])
    axis.text(6.0, 3.28, "答案实体", ha="center", color=COLORS["neutral"])
    axis.text(
        0.0,
        -0.34,
        r"完整判据：$d_I(s,v)+2+d_I(u,a)=d_I(s,a)$；只降低冲突负例的监督强度，不修改标签与推理。",
        fontsize=8,
        color=COLORS["neutral"],
    )
    figure.tight_layout(pad=0.5)
    return save_vector_figure(figure, output_stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
