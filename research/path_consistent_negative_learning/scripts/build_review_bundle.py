"""Assemble the exact, compact reviewer bundle requested for final review."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parents[1]


def _copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _review_snapshot(target: Path, sources: list[Path]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        '"""Reviewer snapshot of production source files; not a standalone module."""',
        "",
    ]
    for source in sources:
        relative = source.relative_to(REPOSITORY).as_posix()
        sections.extend(
            (
                f"# ===== BEGIN PRODUCTION SOURCE: {relative} =====",
                source.read_text(encoding="utf-8").rstrip(),
                f"# ===== END PRODUCTION SOURCE: {relative} =====",
                "",
            )
        )
    target.write_text("\n".join(sections), encoding="utf-8")


def build_bundle(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    paper = PROJECT / "paper" / "www2027"
    docs = PROJECT / "docs"
    artifacts = PROJECT / "artifacts" / "final_revision"
    _copy(paper / "main.pdf", destination / "main.pdf")
    for name in (
        "FINAL_EXPERIMENT_AUDIT.md",
        "EXPERIMENT_PROTOCOL.md",
        "METHOD_DECISION.md",
        "CANDIDATE_ORACLE_ANALYSIS.md",
        "FIXED_MASKING_ANALYSIS.md",
    ):
        _copy(docs / name, destination / name)
    result_sources = {
        "final_main_results.csv": artifacts
        / "fixed_masking"
        / "final_main_results.csv",
        "lambda_validation.csv": artifacts
        / "lambda_validation"
        / "lambda_validation.csv",
        "fixed_masking_results.csv": artifacts
        / "fixed_masking"
        / "fixed_masking_results.csv",
        "candidate_oracle.csv": artifacts
        / "candidate_oracle"
        / "candidate_oracle.csv",
    }
    for name, source in result_sources.items():
        _copy(source, destination / "results" / name)
    _review_snapshot(
        destination / "core_code" / "path_consistency.py",
        [PROJECT / "path_supervision" / "path_consistency.py"],
    )
    _review_snapshot(
        destination / "core_code" / "train_or_loss.py",
        [
            PROJECT / "path_supervision" / "weighted_loss.py",
            PROJECT / "retriever_only" / "training.py",
        ],
    )
    _review_snapshot(
        destination / "core_code" / "candidate_generation.py",
        [
            PROJECT / "retriever_only" / "candidates.py",
            PROJECT / "retriever_only" / "preparation.py",
        ],
    )
    _review_snapshot(
        destination / "core_code" / "evaluator.py",
        [
            PROJECT / "retriever_only" / "metrics.py",
            PROJECT / "retriever_only" / "evaluation.py",
        ],
    )
    expected = {
        "main.pdf",
        "FINAL_EXPERIMENT_AUDIT.md",
        "EXPERIMENT_PROTOCOL.md",
        "METHOD_DECISION.md",
        "CANDIDATE_ORACLE_ANALYSIS.md",
        "FIXED_MASKING_ANALYSIS.md",
        "results/final_main_results.csv",
        "results/lambda_validation.csv",
        "results/fixed_masking_results.csv",
        "results/candidate_oracle.csv",
        "core_code/path_consistency.py",
        "core_code/train_or_loss.py",
        "core_code/candidate_generation.py",
        "core_code/evaluator.py",
    }
    actual = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise RuntimeError(
            f"review bundle contents differ: missing={expected - actual}, "
            f"extra={actual - expected}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=REPOSITORY / "review_bundle",
    )
    args = parser.parse_args()
    build_bundle(args.destination.resolve())
    print(f"review bundle -> {args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
