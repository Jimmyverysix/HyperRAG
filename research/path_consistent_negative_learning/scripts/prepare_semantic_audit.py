"""从路径一致采样负例生成双人盲化语义审计表。"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
import random
from typing import Any


def _trusted_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - 官方数据集格式


def _inverse(mapping: dict[Any, int]) -> dict[int, str]:
    return {int(value): str(key) for key, value in mapping.items()}


def _sample_domain(path: Path, count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    reservoir = []
    eligible = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            unique = {
                (
                    int(candidate["head_id"]),
                    int(candidate["hyperedge_owner_id"]),
                    int(candidate["tail_id"]),
                )
                for sample in row["sampling"]
                for candidate in sample["disputed_samples"]
            }
            if not unique:
                continue
            transition = sorted(unique)[rng.randrange(len(unique))]
            item = {
                "domain": row["domain"],
                "query_index": int(row["query_index"]),
                "topic_id": int(row["query"]["topic_id"]),
                "query_relation_ids": [int(value) for value in row["query"]["relation_ids"]],
                "answer_ids": [int(value) for value in row["reachable_answer_ids"]],
                "head_id": transition[0],
                "hyperedge_owner_id": transition[1],
                "tail_id": transition[2],
            }
            eligible += 1
            if len(reservoir) < count:
                reservoir.append(item)
            else:
                replacement = rng.randrange(eligible)
                if replacement < count:
                    reservoir[replacement] = item
    if len(reservoir) != count:
        raise ValueError(f"{path.stem} 只有 {len(reservoir)} 个可审计问题，少于 {count}")
    return reservoir


def _qid(entity_id: int, mapping: dict[int, str]) -> str:
    return mapping.get(entity_id, f"UNKNOWN_ENTITY_{entity_id}")


def _pid(relation_id: int, mapping: dict[int, str]) -> str:
    return mapping.get(relation_id, f"UNKNOWN_RELATION_{relation_id}")


def _annotation_row(
    item: dict[str, Any],
    entity_mapping: dict[int, str],
    relation_mapping: dict[int, str],
    hyperedge_relations: dict[int, tuple[int, ...]],
) -> dict[str, Any]:
    topic = _qid(item["topic_id"], entity_mapping)
    answers = [_qid(value, entity_mapping) for value in item["answer_ids"]]
    head = _qid(item["head_id"], entity_mapping)
    owner = _qid(item["hyperedge_owner_id"], entity_mapping)
    tail = _qid(item["tail_id"], entity_mapping)
    query_relations = [
        _pid(value, relation_mapping) for value in item["query_relation_ids"]
    ]
    fact_relations = [
        _pid(value, relation_mapping)
        for value in hyperedge_relations.get(item["hyperedge_owner_id"], ())
    ]
    return {
        "item_id": f"{item['domain']}-{item['query_index']}-{item['head_id']}-{item['hyperedge_owner_id']}-{item['tail_id']}",
        "domain": item["domain"],
        "topic_entity": topic,
        "query_relations": ";".join(query_relations),
        "answer_entities": ";".join(answers),
        "candidate_head": head,
        "candidate_fact_owner": owner,
        "candidate_tail": tail,
        "candidate_fact_relations": ";".join(fact_relations),
        "wikidata_urls": ";".join(
            f"https://www.wikidata.org/wiki/{value}"
            for value in dict.fromkeys([topic, *answers, head, owner, tail])
            if value.startswith("Q")
        ),
        "semantic_label": "",
        "confidence_1_to_3": "",
        "rationale": "",
    }


def prepare(
    raw_audit_root: Path,
    dataset_root: Path,
    output_dir: Path,
    *,
    domains: list[str],
    per_domain: int,
    seed: int,
) -> list[dict[str, Any]]:
    all_rows = []
    for domain_index, domain in enumerate(domains):
        items = _sample_domain(
            raw_audit_root / f"{domain}.audit.jsonl",
            per_domain,
            seed + domain_index,
        )
        mapping_payload = _trusted_pickle(dataset_root / domain / "og_mappings.pkl")
        entity_mapping = _inverse(mapping_payload["e2id_train"])
        relation_mapping = _inverse(mapping_payload["r2id"])

        hyperedge_relations: dict[int, set[int]] = {}
        with (dataset_root / domain / "directed" / "train_graph.txt").open(
            "r", encoding="utf-8"
        ) as handle:
            for line in handle:
                head, relation, _ = (int(value) for value in line.split())
                hyperedge_relations.setdefault(head, set()).add(relation)
        normalized_relations = {
            key: tuple(sorted(values)) for key, values in hyperedge_relations.items()
        }
        all_rows.extend(
            _annotation_row(
                item,
                entity_mapping,
                relation_mapping,
                normalized_relations,
            )
            for item in items
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0])
    for annotator_index, filename in enumerate(("annotator_a.csv", "annotator_b.csv")):
        rows = list(all_rows)
        random.Random(seed + 10_000 + annotator_index).shuffle(rows)
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "sampling_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "domains": domains,
                "per_domain": per_domain,
                "seed": seed,
                "item_count": len(all_rows),
                "annotation_files": ["annotator_a.csv", "annotator_b.csv"],
                "labels": ["relevant", "irrelevant", "unclear"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-audit-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--per-domain", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260920)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = prepare(
        args.raw_audit_root,
        args.dataset_root,
        args.output_dir,
        domains=args.domains,
        per_domain=args.per_domain,
        seed=args.seed,
    )
    print(f"已生成 {len(rows)} 个盲化语义审计条目；等待两名真实标注者。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
