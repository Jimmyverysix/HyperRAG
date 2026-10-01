"""Generate the three formal, deterministic Retriever-only paper figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np

from .www_style import COLORS, apply_style, save_vector_figure


CONFLICT_COLOR = "#D55E00"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _shorten(text: str, limit: int = 20) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _draw_path(
    axis,
    nodes: Sequence[Mapping[str, str]],
    *,
    y: float,
    color: str,
    highlighted: tuple[str, str, str] | None,
) -> None:
    x_values = np.linspace(0.08, 0.92, len(nodes))
    highlighted_ids = set(highlighted or ())
    for index in range(len(nodes) - 1):
        first = nodes[index]["node_id"]
        second = nodes[index + 1]["node_id"]
        edge_color = (
            CONFLICT_COLOR
            if first in highlighted_ids and second in highlighted_ids
            else color
        )
        axis.plot(
            x_values[index : index + 2],
            [y, y],
            color=edge_color,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=1,
        )
    for x_value, node in zip(x_values, nodes):
        is_hyperedge = node["kind"] == "hyperedge"
        face = "#F4F6F7" if is_hyperedge else "white"
        edge = CONFLICT_COLOR if node["node_id"] in highlighted_ids else color
        if is_hyperedge:
            patch = FancyBboxPatch(
                (x_value - 0.035, y - 0.045),
                0.07,
                0.09,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.7,
                zorder=2,
            )
        else:
            patch = Circle(
                (x_value, y),
                0.035,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.7,
                zorder=2,
            )
        axis.add_patch(patch)
        label = _shorten(node["text"].split(" | ", 1)[0], 16)
        axis.text(x_value, y - 0.075, label, ha="center", va="top", fontsize=6.8)


def generate_method_figure(case: Mapping[str, Any], output_dir: Path) -> None:
    apply_style()
    selected = case["selected_path_for_witness_answer"]
    witness = case["conflict_witness_path"]
    transition = case["sampled_negative_transition"]
    highlighted = (
        transition["head"],
        transition["hyperedge"],
        transition["tail"],
    )
    figure, axis = plt.subplots(figsize=(7.1, 2.6))
    _draw_path(axis, selected, y=0.69, color=COLORS["ours"], highlighted=None)
    _draw_path(
        axis,
        witness,
        y=0.29,
        color="#7A7A7A",
        highlighted=highlighted,
    )
    axis.text(0.01, 0.81, "被选中的最短路径：正监督", color=COLORS["ours"])
    axis.text(0.01, 0.41, "另一条等长最短路径：其中一段被采样为负例")
    equality = case["distance_equality"]
    formula = (
        f"{equality['topic_to_head']} + {equality['transition_cost']} + "
        f"{equality['tail_to_answer']} = {equality['topic_to_answer']}"
    )
    axis.text(
        0.99,
        0.05,
        f"完整路径条件：{formula}",
        ha="right",
        fontsize=7.5,
        color=CONFLICT_COLOR,
    )
    axis.text(
        0.01,
        0.98,
        "真实问题：" + _shorten(str(case["question"]), 85),
        va="top",
        fontsize=8.2,
    )
    axis.text(
        0.5,
        0.50,
        "只降低橙色负例的损失权重；不把它改成语义正例",
        ha="center",
        fontsize=8,
        fontweight="bold",
        color=CONFLICT_COLOR,
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    save_vector_figure(figure, output_dir / "supervision_conflict")


def generate_prevalence_figure(
    prevalence: Mapping[str, Any],
    output_dir: Path,
) -> None:
    apply_style()
    rows = prevalence["domains"]
    domains = [row["domain"] for row in rows]
    conflict = np.asarray(
        [100.0 * float(row["path_consistent_negative_rate"]) for row in rows]
    )
    affected = np.asarray(
        [100.0 * float(row["affected_query_rate_across_seeds"]) for row in rows]
    )
    order = np.argsort(affected)
    domains = [domains[index] for index in order]
    conflict = conflict[order]
    affected = affected[order]
    y_values = np.arange(len(domains))
    figure, axis = plt.subplots(figsize=(7.1, 3.7))
    axis.barh(
        y_values - 0.18,
        conflict,
        height=0.34,
        color=CONFLICT_COLOR,
        label="采样负例中的冲突比例",
    )
    axis.barh(
        y_values + 0.18,
        affected,
        height=0.34,
        color=COLORS["ours_soft"],
        label="至少一个种子受影响的问题比例",
    )
    for y_value, value in zip(y_values, conflict):
        axis.text(
            value + 0.45,
            y_value - 0.18,
            f"{value:.2f}",
            va="center",
            fontsize=6.6,
        )
    for y_value, value in zip(y_values, affected):
        axis.text(
            value + 0.45,
            y_value + 0.18,
            f"{value:.1f}",
            va="center",
            fontsize=6.6,
        )
    axis.set_yticks(y_values, domains)
    axis.set_xlabel("比例（%）")
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right")
    figure.tight_layout()
    save_vector_figure(figure, output_dir / "conflict_prevalence")


def generate_main_figure(
    main: Mapping[str, Any],
    output_dir: Path,
) -> None:
    """Draw per-domain APC-MRR effects with paired 95% intervals."""

    apply_style()
    lookup = {
        (row["domain"], row["reference"]): row
        for row in main["domain_comparisons"]
    }
    domains = [row["domain"] for row in main["domains"]]
    order = sorted(
        range(len(domains)),
        key=lambda index: lookup[(domains[index], "baseline")][
            "difference_percentage_points"
        ],
    )
    domains = [domains[index] for index in order]
    y_values = np.arange(len(domains))
    figure, axis = plt.subplots(figsize=(7.1, 4.2))
    axis.axvline(0.0, color="#777777", linewidth=0.8)
    for offset, reference, color, marker, label in (
        (-0.14, "baseline", COLORS["ours"], "o", "策略3 − 策略1"),
        (0.14, "matched_random", COLORS["random"], "s", "策略3 − 策略2"),
    ):
        rows = [lookup[(domain, reference)] for domain in domains]
        centers = np.asarray(
            [float(row["difference_percentage_points"]) for row in rows]
        )
        lower = centers - np.asarray(
            [float(row["ci_low_percentage_points"]) for row in rows]
        )
        upper = np.asarray(
            [float(row["ci_high_percentage_points"]) for row in rows]
        ) - centers
        axis.errorbar(
            centers,
            y_values + offset,
            xerr=np.vstack((lower, upper)),
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.0,
            capsize=2.0,
            linestyle="none",
            label=label,
        )
    axis.set_yticks(y_values, domains)
    axis.set_xlabel("APC-MRR 差值（百分点，95% 配对 bootstrap CI）")
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="best")
    figure.tight_layout()
    save_vector_figure(figure, output_dir / "main_effects")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-study", type=Path, required=True)
    parser.add_argument("--prevalence", type=Path, required=True)
    parser.add_argument("--main-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    generate_method_figure(_load(args.case_study), args.output_dir)
    generate_prevalence_figure(_load(args.prevalence), args.output_dir)
    generate_main_figure(_load(args.main_test), args.output_dir)
    print(f"figures -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
