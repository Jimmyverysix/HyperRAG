"""在 lambda 冻结后，用 selection 阶段检查点评估独立 test split。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from research.path_consistent_negative_learning.structured_data import (
    TEST_SPLIT,
    StructuredExperimentData,
)
from research.path_consistent_negative_learning.structured_model import StructuredRetriever
from research.path_consistent_negative_learning.train_structured import evaluate_split


def evaluate_checkpoint(
    *,
    data_path: Path,
    source_result_path: Path,
    lambda_star_path: Path,
    arm: str,
    output_dir: Path,
    device_name: str,
    batch_size: int,
) -> dict:
    selection = json.loads(lambda_star_path.read_text(encoding="utf-8"))
    if selection.get("test_metrics_accessed") is not False:
        raise ValueError("lambda_star.json 没有通过 test 隔离声明")
    source = json.loads(source_result_path.read_text(encoding="utf-8"))
    allowed = {
        "baseline_lambda_1": 1.0,
        "mask_endpoint_lambda_0": 0.0,
        "dataset_specific_lambda_star": float(selection["best_lambda"]),
    }
    if arm not in allowed:
        raise ValueError(f"未知冻结评估臂：{arm}")
    expected_lambda = allowed[arm]
    if source.get("evaluation_split") != "selection":
        raise ValueError("源检查点必须来自 selection sweep")
    if float(source["disputed_negative_weight"]) != expected_lambda:
        raise ValueError("源检查点的 lambda 与冻结评估臂不一致")
    if source["domain"] != selection["dataset"]:
        raise ValueError("lambda_star dataset 与源检查点不一致")

    data = StructuredExperimentData.load(data_path)
    if data.domain != source["domain"] or data.sampler_seed != source["seed"]:
        raise ValueError("数据文件与源检查点元数据不一致")
    device = torch.device(device_name)
    model = StructuredRetriever(
        entity_count=data.entity_count,
        relation_count=data.relation_count,
        hyperedge_relation_mask=data.hyperedge_relation_mask,
        numeric_feature_count=data.numeric_features.shape[1],
    ).to(device)
    checkpoint = source_result_path.with_name("best_model.pt")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    metrics, query_rows = evaluate_split(
        model,
        data,
        device=device,
        batch_size=batch_size,
        split=TEST_SPLIT,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "experiment": "www_revision_proxy_test",
        "domain": data.domain,
        "arm": arm,
        "strategy": "strategy2_weighted",
        "seed": int(source["seed"]),
        "sampler_seed": data.sampler_seed,
        "split_seed": data.split_seed,
        "split_scheme": data.split_scheme,
        "evaluation_split": "test",
        "disputed_negative_weight": expected_lambda,
        "lambda_star_file": str(lambda_star_path),
        "source_selection_result": str(source_result_path),
        "source_best_epoch": int(source["best_epoch"]),
        "metrics": metrics,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if query_rows:
        with (output_dir / "query_metrics.jsonl").open("w", encoding="utf-8") as handle:
            for row in query_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        with (output_dir / "query_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(query_rows[0]))
            writer.writeheader()
            writer.writerows(query_rows)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--lambda-star", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=(
            "baseline_lambda_1",
            "mask_endpoint_lambda_0",
            "dataset_specific_lambda_star",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = evaluate_checkpoint(
        data_path=args.data,
        source_result_path=args.source_result,
        lambda_star_path=args.lambda_star,
        arm=args.arm,
        output_dir=args.output_dir,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
