"""检查官方 HyperRetriever 各阶段是否具备可执行前置条件。

该脚本只做只读检查，不下载数据、不安装依赖，也不发起 API 请求。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


DOMAINS = (
    "art",
    "award",
    "edu",
    "health",
    "infra",
    "loc",
    "org",
    "people",
    "sci",
    "sport",
    "tax",
)

STAGE_PACKAGES = {
    "graph-build": ("torch", "networkx", "numpy", "openai", "tiktoken"),
    "retriever-train": (
        "torch",
        "networkx",
        "numpy",
        "sklearn",
        "transformers",
    ),
    "qa-eval": ("torch", "networkx", "numpy", "openai", "transformers"),
}

STAGE_DATA_FILES = {
    "graph-build": ("train_sentences.txt", "test_sentences.txt"),
    "retriever-train": ("train_queries.json", "train_answers_hard.json"),
    "qa-eval": ("test_queries.json", "test_answers_easy.json", "test_answers_hard.json"),
}


def _has_api_credentials(repo_root: Path) -> bool:
    if os.environ.get("OPENAI_API_KEY"):
        return True

    config_path = repo_root / "config.json"
    if not config_path.is_file():
        return False

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    return any(config.get(name) for name in ("openai_api_key", "hyperretriever_api_key"))


def _source_checks(repo_root: Path) -> dict[str, bool]:
    required = {
        "wiki_prepare": repo_root / "HyperRetriever/retrieve/prepare.py",
        "wiki_train": repo_root / "HyperRetriever/retrieve/train.py",
        "wiki_query": repo_root / "HyperRetriever/wikitopics_query.py",
        "wiki_evaluator": repo_root / "evaluate/qa_eval_MRR_HIT.py",
    }
    return {name: path.is_file() for name, path in required.items()}


def _data_checks(data_root: Path, stage: str) -> dict[str, bool]:
    filenames = STAGE_DATA_FILES[stage]
    return {
        domain: all((data_root / domain / filename).is_file() for filename in filenames)
        for domain in DOMAINS
    }


def _asset_checks(expr_root: Path, stage: str) -> dict[str, bool]:
    if stage == "graph-build":
        return {}

    required = ["graph_chunk_entity_relation.graphml"]
    if stage == "qa-eval":
        required.extend(("graph_embedding_nodes.json", "graph_embeddings.pt"))

    checks: dict[str, bool] = {}
    for domain in DOMAINS:
        domain_root = expr_root / "wikitopics" / domain
        files_present = all((domain_root / filename).is_file() for filename in required)
        if stage == "qa-eval":
            files_present = files_present and (
                domain_root / "train/best_retrieval_model.pth"
            ).is_file()
        checks[domain] = files_present
    return checks


def _package_checks(stage: str) -> dict[str, bool]:
    return {
        package: importlib.util.find_spec(package) is not None
        for package in STAGE_PACKAGES[stage]
    }


def inspect_prerequisites(
    repo_root: Path,
    data_root: Path,
    expr_root: Path,
    stage: str,
) -> dict[str, Any]:
    source = _source_checks(repo_root)
    data = _data_checks(data_root, stage)
    assets = _asset_checks(expr_root, stage)
    packages = _package_checks(stage)
    api_credentials = _has_api_credentials(repo_root)

    groups = (source, data, packages)
    if assets:
        groups = groups + (assets,)
    ready = all(all(group.values()) for group in groups) and api_credentials

    return {
        "stage": stage,
        "ready": ready,
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "expr_root": str(expr_root),
        "checks": {
            "source_files": source,
            "dataset_domains": data,
            "runtime_packages": packages,
            "stage_assets": assets,
            "api_credentials_present": api_credentials,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_PACKAGES),
        required=True,
        help="要检查的官方流程阶段。",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/WikiTopicsQE_NLG"),
    )
    parser.add_argument("--expr-root", type=Path, default=Path("expr"))
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect_prerequisites(
        repo_root=args.repo_root.resolve(),
        data_root=args.data_root.resolve(),
        expr_root=args.expr_root.resolve(),
        stage=args.stage,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")

    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
