"""为 WWW 修订版生成共享候选、问题级划分的结构代理数据。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from research.path_consistent_negative_learning.structured_data import (
    StructuredExperimentData,
    build_structured_experiment_data,
)


def _prepare_one(
    domain_dir: Path,
    output: Path,
    *,
    seed: int,
    split_seed: int,
    split_scheme: str,
) -> str:
    if output.is_file():
        data = StructuredExperimentData.load(output)
        expected = (domain_dir.name, seed, split_seed, split_scheme)
        actual = (data.domain, data.sampler_seed, data.split_seed, data.split_scheme)
        if actual != expected:
            raise ValueError(f"已有数据与协议不一致：{output}: {actual} != {expected}")
        return f"skipped {data.domain} seed={seed}"
    data = build_structured_experiment_data(
        domain_dir,
        sampler_seed=seed,
        split_seed=split_seed,
        split_scheme=split_scheme,
    )
    data.save(output)
    return f"prepared {data.domain} seed={seed}"


def prepare(config_path: Path, source_root: Path, output_root: Path, workers: int) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = config["randomness"]["sampler_and_training_seeds"]
    split_seed = config["randomness"]["split_seed"]
    split_scheme = config["randomness"]["split_scheme"]
    tasks = []
    for domain in config["datasets"]:
        # WikiTopics_QE stores the integer query/answer mappings and the
        # training graph together at the domain root.  The nested ``directed``
        # directory contains inference exports only and is not a loadable
        # training dataset on its own.
        domain_dir = source_root / domain
        if not domain_dir.is_dir():
            raise FileNotFoundError(domain_dir)
        for seed in seeds:
            tasks.append(
                (
                    domain_dir,
                    output_root / domain / f"seed_{seed}.pt",
                    int(seed),
                )
            )
    if not 1 <= workers <= 6:
        raise ValueError("workers 必须位于 1--6")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "COMPLETE").unlink(missing_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _prepare_one,
                domain_dir,
                output,
                seed=seed,
                split_seed=split_seed,
                split_scheme=split_scheme,
            ): (domain_dir.name, seed)
            for domain_dir, output, seed in tasks
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)
    (output_root / "COMPLETE").write_text("all data prepared\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prepare(args.config, args.source_root, args.output_root, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
