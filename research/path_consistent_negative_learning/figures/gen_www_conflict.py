"""绘制候选空间与实际采样中的路径冲突富集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .www_style import COLORS, apply_style, panel_label, save_vector_figure


def generate(audit_path: Path, output_stem: Path) -> tuple[Path, Path]:
    apply_style()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rows = audit["domains"]
    domains = [row["domain"] for row in rows]
    x = np.arange(len(rows))
    candidate = np.asarray([row["candidate_conflict_rate"] for row in rows]) * 100
    sampled = np.asarray([row["sampled_conflict_rate"] for row in rows]) * 100
    candidate_questions = np.asarray(
        [row["queries_with_candidate_conflict_rate"] for row in rows]
    ) * 100
    sampled_questions = np.asarray(
        [row["queries_with_sampled_conflict_any_seed_rate"] for row in rows]
    ) * 100

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.05),
        layout="constrained",
        gridspec_kw={"wspace": 0.16},
    )
    ax = axes[0]
    ax.plot(
        x,
        candidate,
        color=COLORS["baseline"],
        marker="o",
        markersize=3.8,
        linewidth=1.4,
        label="完整候选空间",
    )
    ax.plot(
        x,
        sampled,
        color=COLORS["ours"],
        marker="o",
        markersize=3.8,
        linewidth=1.6,
        label="实际采样负例",
    )
    ax.set_yscale("log")
    ax.set_ylabel("路径一致负例比例（%）")
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha="right")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=7,
    )
    panel_label(ax, "a")

    ax = axes[1]
    width = 0.38
    ax.bar(
        x - width / 2,
        candidate_questions,
        width,
        color=COLORS["baseline"],
        label="候选空间至少一个",
    )
    ax.bar(
        x + width / 2,
        sampled_questions,
        width,
        color=COLORS["ours"],
        label="五个 seed 至少采到一次",
    )
    ax.set_ylabel("问题占比（%）")
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha="right")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=7,
    )
    panel_label(ax, "b")
    return save_vector_figure(figure, output_stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    generate(args.audit, args.output_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
