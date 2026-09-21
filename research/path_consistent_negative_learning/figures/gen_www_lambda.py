"""绘制 lambda 敏感性、逐领域选择与冲突率关系。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from .www_style import COLORS, apply_style, panel_label, save_vector_figure


def generate(
    selection_csv: Path, audit_path: Path, output_stem: Path
) -> tuple[Path, Path]:
    apply_style()
    with selection_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    conflict = {
        row["domain"]: float(row["sampled_conflict_rate"]) * 100
        for row in audit["domains"]
    }
    lambdas = np.asarray([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    score_columns = [f"lambda_{value:g}" for value in lambdas]
    score_matrix = np.asarray(
        [[float(row[column]) for column in score_columns] for row in rows]
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.5, 2.75),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.35, 0.95, 1.0], "wspace": 0.16},
    )
    ax = axes[0]
    for row_scores in score_matrix:
        ax.plot(lambdas, row_scores, color=COLORS["ours_soft"], linewidth=0.8, alpha=0.65)
    ax.plot(
        lambdas,
        score_matrix.mean(axis=0),
        color=COLORS["ours"],
        marker="o",
        markersize=4,
        linewidth=2,
        label="领域宏平均",
    )
    ax.set_xlabel("路径一致负例权重 λ")
    ax.set_ylabel("selection answer reach@10")
    ax.set_xticks(lambdas)
    ax.legend(loc="best")
    panel_label(ax, "a")

    selected = np.asarray([float(row["best_lambda"]) for row in rows])
    order = np.argsort(selected)
    ax = axes[1]
    ax.scatter(selected[order], np.arange(len(rows)), color=COLORS["ours"], s=23)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([rows[index]["dataset"] for index in order])
    ax.set_xticks(lambdas)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.set_xlabel("冻结的 λ*")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    panel_label(ax, "b")

    rates = np.asarray([conflict[row["dataset"]] for row in rows])
    correlation = spearmanr(rates, selected)
    ax = axes[2]
    ax.scatter(rates, selected, color=COLORS["ours"], s=26, edgecolor="white", linewidth=0.5)
    for x_value, y_value, row in zip(rates, selected, rows):
        if not np.isclose(y_value, 0.0):
            ax.annotate(
                row["dataset"],
                (x_value, y_value),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=6,
            )
    zero_count = int(np.isclose(selected, 0.0).sum())
    ax.text(
        0.97,
        0.35,
        f"其余 {zero_count} 个领域：λ*=0",
        transform=ax.transAxes,
        ha="right",
        color=COLORS["neutral"],
        fontsize=7,
    )
    ax.set_xlabel("采样冲突率（%）")
    ax.set_ylabel("冻结的 λ*")
    ax.set_yticks(lambdas)
    ax.text(
        0.03,
        0.97,
        f"Spearman ρ={correlation.statistic:.2f}\np={correlation.pvalue:.3f}",
        transform=ax.transAxes,
        va="top",
        color=COLORS["neutral"],
    )
    panel_label(ax, "c")
    return save_vector_figure(figure, output_stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    generate(args.selection_csv, args.audit, args.output_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
