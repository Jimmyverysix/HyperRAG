"""Command-line stages for the zero-LLM Retriever-only experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

from research.path_consistent_negative_learning.retriever_only.data import (
    LabelSnapshot,
    load_aligned_queries,
)
from research.path_consistent_negative_learning.retriever_only.embeddings import (
    EmbeddingStore,
    OfficialGTEEncoder,
    encode_store,
)
from research.path_consistent_negative_learning.retriever_only.evaluation import (
    candidate_path_coverage,
    evaluate_scores,
    score_candidates,
)
from research.path_consistent_negative_learning.retriever_only.graph import (
    build_deterministic_hypergraph,
)
from research.path_consistent_negative_learning.retriever_only.official import OfficialDDE
from research.path_consistent_negative_learning.retriever_only.preparation import (
    prepare_evaluation_candidates,
    prepare_training_candidates,
)
from research.path_consistent_negative_learning.retriever_only.prepared import (
    PreparedCandidates,
)
from research.path_consistent_negative_learning.retriever_only.provenance import (
    collect_provenance,
)
from research.path_consistent_negative_learning.retriever_only.training import (
    TrainingConfig,
    train_retriever,
)


REPOSITORY = Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _domain_inputs(args: argparse.Namespace):
    labels = LabelSnapshot.load(args.label_snapshot)
    structured_dir = args.structured_root / args.domain
    nlg_dir = args.nlg_root / args.domain
    bundle = build_deterministic_hypergraph(structured_dir, labels)
    return labels, structured_dir, nlg_dir, bundle


def command_encode(args: argparse.Namespace) -> dict[str, Any]:
    labels, structured_dir, nlg_dir, bundle = _domain_inputs(args)
    queries = [
        query
        for split in ("train", "valid", "test")
        for query in load_aligned_queries(structured_dir, nlg_dir, split, labels)
    ]
    encoder = OfficialGTEEncoder(args.model_path, args.device)
    store = encode_store(
        bundle.node_texts,
        [query.text for query in queries],
        encoder,
        batch_size=args.batch_size,
    )
    store.save(args.output)
    return {
        "stage": "encode",
        "domain": args.domain,
        "node_count": len(store.node_names),
        "unique_query_count": len(store.query_texts),
        "output": str(args.output),
    }


def command_prepare_train(args: argparse.Namespace) -> dict[str, Any]:
    labels, structured_dir, nlg_dir, bundle = _domain_inputs(args)
    queries = load_aligned_queries(structured_dir, nlg_dir, "train", labels)
    store = EmbeddingStore.load(args.embeddings)
    dde = OfficialDDE(device=args.device)
    data = prepare_training_candidates(
        args.domain,
        queries,
        bundle,
        store,
        dde,
        seed=args.seed,
        variant_seed=args.variant_seed,
    )
    data.save(args.output)
    return {
        "stage": "prepare_train",
        "domain": args.domain,
        "seed": args.seed,
        "variant_seed": args.variant_seed,
        "query_count": len(data.query_keys),
        "candidate_count": len(data.dde_features),
        "positive_count": int(data.labels.sum()),
        "path_consistent_negative_count": int(data.path_consistent_mask.sum()),
        "output": str(args.output),
    }


def _require_selection(args: argparse.Namespace) -> dict[str, Any]:
    if args.selection_file is None or not args.selection_file.is_file():
        raise ValueError("test access requires an existing frozen selection file")
    return json.loads(args.selection_file.read_text(encoding="utf-8"))


def command_prepare_eval(args: argparse.Namespace) -> dict[str, Any]:
    if args.split == "test":
        _require_selection(args)
    labels, structured_dir, nlg_dir, bundle = _domain_inputs(args)
    queries = load_aligned_queries(structured_dir, nlg_dir, args.split, labels)
    store = EmbeddingStore.load(args.embeddings)
    dde = OfficialDDE(device=args.device)
    data = prepare_evaluation_candidates(
        args.domain,
        args.split,
        queries,
        bundle,
        store,
        dde,
        similarity_device=args.device,
        beam_width=args.beam_width,
    )
    data.save(args.output)
    return {
        "stage": "prepare_eval",
        "domain": args.domain,
        "split": args.split,
        "beam_width": args.beam_width,
        "query_count": len(data.query_keys),
        "candidate_count": len(data.dde_features),
        "candidate_path_coverage": candidate_path_coverage(data),
        "output": str(args.output),
    }


def command_train(args: argparse.Namespace) -> dict[str, Any]:
    data = PreparedCandidates.load(args.training_data)
    if data.seed != args.seed:
        raise ValueError("training data seed does not match the run seed")
    if args.method == "baseline" and args.lambda_ != 1.0:
        raise ValueError("baseline must use lambda=1.0")
    result = train_retriever(
        data,
        EmbeddingStore.load(args.embeddings),
        method=args.method,
        lambda_=args.lambda_,
        seed=args.seed,
        device_name=args.device,
        checkpoint_path=args.checkpoint,
        config=TrainingConfig(),
    )
    result.update(
        {
            "stage": "train",
            "checkpoint": str(args.checkpoint),
            "training_data": str(args.training_data),
        }
    )
    return result


def command_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    data = PreparedCandidates.load(args.evaluation_data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    evaluation_arm = args.evaluation_arm or checkpoint["method"]
    if data.split == "test":
        selection = _require_selection(args)
        selected = float(selection["domains"][data.domain]["lambda"])
        if evaluation_arm in {"fixed_masking", "fixed_matched_random"}:
            if float(checkpoint["lambda"]) != 0.0:
                raise ValueError("fixed-masking post-hoc evaluation requires lambda=0.0")
            expected_method = (
                "ours" if evaluation_arm == "fixed_masking" else "matched_random"
            )
            if checkpoint["method"] != expected_method:
                raise ValueError("fixed-masking checkpoint method mismatch")
        elif evaluation_arm == "ours" and float(checkpoint["lambda"]) != selected:
            raise ValueError("checkpoint lambda does not match frozen selection")
        elif (
            evaluation_arm == "matched_random"
            and float(checkpoint["lambda"]) != selected
        ):
            raise ValueError(
                "matched-random checkpoint lambda does not match frozen selection"
            )
        if evaluation_arm == "baseline" and float(checkpoint["lambda"]) != 1.0:
            raise ValueError("baseline evaluation requires the lambda=1.0 checkpoint")
    scores = score_candidates(
        data,
        EmbeddingStore.load(args.embeddings),
        args.checkpoint,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scores, args.scores)
    result = evaluate_scores(
        data,
        scores,
        multiple_shortest_only=args.multiple_shortest_only,
    )
    result.update(
        {
            "stage": "evaluate",
            "method": evaluation_arm,
            "checkpoint_training_method": checkpoint["method"],
            "lambda": checkpoint["lambda"],
            "seed": checkpoint["seed"],
            "checkpoint": str(args.checkpoint),
            "evaluation_data": str(args.evaluation_data),
            "scores": str(args.scores),
            "analysis_design": (
                "POST-HOC SIMPLIFICATION ANALYSIS"
                if evaluation_arm in {"fixed_masking", "fixed_matched_random"}
                else "confirmatory frozen protocol"
            ),
        }
    )
    return result


def _common_data(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--structured-root", type=Path, required=True)
    parser.add_argument("--nlg-root", type=Path, required=True)
    parser.add_argument("--label-snapshot", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--device", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    encode = subparsers.add_parser("encode")
    _common_data(encode)
    encode.add_argument("--model-path", type=Path, required=True)
    encode.add_argument("--batch-size", type=int, default=32)
    encode.add_argument("--output", type=Path, required=True)
    encode.set_defaults(handler=command_encode)

    prepare_train = subparsers.add_parser("prepare-train")
    _common_data(prepare_train)
    prepare_train.add_argument("--embeddings", type=Path, required=True)
    prepare_train.add_argument("--seed", type=int, required=True)
    prepare_train.add_argument("--variant-seed", type=int)
    prepare_train.add_argument("--output", type=Path, required=True)
    prepare_train.set_defaults(handler=command_prepare_train)

    prepare_eval = subparsers.add_parser("prepare-eval")
    _common_data(prepare_eval)
    prepare_eval.add_argument("--embeddings", type=Path, required=True)
    prepare_eval.add_argument("--split", choices=("valid", "test"), required=True)
    prepare_eval.add_argument("--beam-width", type=int, default=10)
    prepare_eval.add_argument("--selection-file", type=Path)
    prepare_eval.add_argument("--output", type=Path, required=True)
    prepare_eval.set_defaults(handler=command_prepare_eval)

    train = subparsers.add_parser("train")
    train.add_argument("--training-data", type=Path, required=True)
    train.add_argument("--embeddings", type=Path, required=True)
    train.add_argument(
        "--method", choices=("baseline", "ours", "matched_random"), required=True
    )
    train.add_argument("--lambda", dest="lambda_", type=float, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device", required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.set_defaults(handler=command_train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--evaluation-data", type=Path, required=True)
    evaluate.add_argument("--embeddings", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument(
        "--evaluation-arm",
        choices=(
            "baseline",
            "ours",
            "matched_random",
            "fixed_masking",
            "fixed_matched_random",
        ),
    )
    evaluate.add_argument("--device", required=True)
    evaluate.add_argument("--batch-size", type=int, default=1024)
    evaluate.add_argument("--selection-file", type=Path)
    evaluate.add_argument("--multiple-shortest-only", action="store_true")
    evaluate.add_argument("--scores", type=Path, required=True)
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = args.handler(args)
    result["provenance"] = collect_provenance(REPOSITORY, sys.argv)
    report_path = getattr(args, "report", None)
    if report_path is None:
        output = getattr(args, "output", None)
        report_path = (
            output.with_suffix(output.suffix + ".report.json")
            if output is not None
            else args.scores.with_suffix(".report.json")
            if hasattr(args, "scores")
            else args.checkpoint.with_suffix(".report.json")
        )
    _write_json(report_path, result)
    print(json.dumps({key: value for key, value in result.items() if key != "queries"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
