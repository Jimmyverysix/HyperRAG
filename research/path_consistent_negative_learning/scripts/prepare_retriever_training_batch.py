"""Prepare several Retriever training seeds with shared structural work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from research.path_consistent_negative_learning.retriever_only.data import (
    LabelSnapshot,
    load_aligned_queries,
)
from research.path_consistent_negative_learning.retriever_only.embeddings import (
    EmbeddingStore,
)
from research.path_consistent_negative_learning.retriever_only.graph import (
    build_deterministic_hypergraph,
)
from research.path_consistent_negative_learning.retriever_only.official import (
    OfficialDDE,
)
from research.path_consistent_negative_learning.retriever_only.preparation import (
    prepare_training_candidates_multi,
)
from research.path_consistent_negative_learning.retriever_only.provenance import (
    collect_provenance,
)


REPOSITORY = Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-root", type=Path, required=True)
    parser.add_argument("--nlg-root", type=Path, required=True)
    parser.add_argument("--label-snapshot", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--variant-seed", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--outputs", type=Path, nargs="+", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def _complete_pair(output: Path) -> bool:
    return output.is_file() and Path(str(output) + ".report.json").is_file()


def main() -> int:
    args = build_parser().parse_args()
    if len(args.seeds) != len(args.outputs):
        raise ValueError("--seeds and --outputs must have the same length")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must be unique")

    requested = dict(zip(args.seeds, args.outputs))
    incomplete: dict[int, Path] = {}
    skipped: list[int] = []
    for seed, output in requested.items():
        report = Path(str(output) + ".report.json")
        if _complete_pair(output):
            skipped.append(seed)
        elif output.exists() or report.exists():
            raise RuntimeError(
                "refusing to overwrite an incomplete historical pair: "
                f"{output}, {report}"
            )
        else:
            incomplete[seed] = output

    provenance = collect_provenance(REPOSITORY, sys.argv)
    results: dict[str, Any] = {}
    if incomplete:
        labels = LabelSnapshot.load(args.label_snapshot)
        structured_dir = args.structured_root / args.domain
        nlg_dir = args.nlg_root / args.domain
        bundle = build_deterministic_hypergraph(structured_dir, labels)
        queries = load_aligned_queries(structured_dir, nlg_dir, "train", labels)
        store = EmbeddingStore.load(args.embeddings)
        prepared = prepare_training_candidates_multi(
            args.domain,
            queries,
            bundle,
            store,
            OfficialDDE(device=args.device),
            seeds=tuple(incomplete),
            variant_seed=args.variant_seed,
        )
        for seed, output in incomplete.items():
            data = prepared[seed]
            data.save(output)
            report = {
                "stage": "prepare_train",
                "batch_preparation": True,
                "domain": args.domain,
                "seed": seed,
                "variant_seed": args.variant_seed,
                "query_count": len(data.query_keys),
                "candidate_count": len(data.dde_features),
                "positive_count": int(data.labels.sum()),
                "path_consistent_negative_count": int(
                    data.path_consistent_mask.sum()
                ),
                "output": str(output),
                "provenance": provenance,
            }
            _write_json(Path(str(output) + ".report.json"), report)
            results[str(seed)] = report

    summary = {
        "stage": "prepare_train_batch",
        "domain": args.domain,
        "variant_seed": args.variant_seed,
        "requested_seeds": list(args.seeds),
        "completed_seeds": sorted(incomplete),
        "skipped_existing_seeds": sorted(skipped),
        "outputs": {str(seed): str(output) for seed, output in requested.items()},
        "provenance": provenance,
    }
    _write_json(args.report, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
