"""绘制 tuned weak supervision 的 coverage--ranking 权衡与配对区间。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .www_style import COLORS, apply_style, panel_label, save_vector_figure


def generate(summary_path: Path, output_stem: Path) -> tuple[Path, Path]:
    apply_style()
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    domains = list(report["domains"])
    reach_delta = []
    precision_delta = []
    ci_low = []
    ci_high = []
    for domain in domains:
        comparisons = report["domains"][domain]["comparisons"]
        reach = comparisons[
            "dataset_specific_lambda_star_vs_baseline_lambda_1_answer_reach_10"
        ]["paired_bootstrap"]
        precision = comparisons[
            "dataset_specific_lambda_star_vs_baseline_lambda_1_selected_pr_auc"
        ]["paired_bootstrap"]
        reach_delta.append(100 * reach["mean_difference"])
        precision_delta.append(100 * precision["mean_difference"])
        ci_low.append(100 * reach["ci_low"])
        ci_high.append(100 * reach["ci_high"])

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    colors = [COLORS["gain"] if value >= 0 else COLORS["drop"] for value in precision_delta]
    ax.scatter(reach_delta, precision_delta, c=colors, s=29, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=COLORS["neutral"], linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.8, linestyle="--", alpha=0.7)
    for x_value, y_value, domain in zip(reach_delta, precision_delta, domains):
        ax.annotate(domain, (x_value, y_value), xytext=(3, 2), textcoords="offset points", fontsize=6)
    ax.set_xlabel("Δ answer reach@10（百分点）")
    ax.set_ylabel("Δ selected-path PR-AUC（百分点）")
    panel_label(ax, "a")

    order = np.argsort(reach_delta)
    y = np.arange(len(domains))
    means = np.asarray(reach_delta)[order]
    lows = np.asarray(ci_low)[order]
    highs = np.asarray(ci_high)[order]
    ax = axes[1]
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([means - lows, highs - means]),
        fmt="o",
        color=COLORS["ours"],
        ecolor=COLORS["ours_soft"],
        capsize=2.5,
        markersize=4,
        linewidth=1.1,
    )
    ax.axvline(0, color=COLORS["neutral"], linewidth=0.9, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([domains[index] for index in order])
    ax.set_xlabel("tuned − baseline answer reach@10（百分点，95% CI）")
    ax.set_ylim(-0.7, len(domains) - 0.3)
    panel_label(ax, "b")
    figure.tight_layout(pad=0.8)
    return save_vector_figure(figure, output_stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    generate(args.summary, args.output_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
