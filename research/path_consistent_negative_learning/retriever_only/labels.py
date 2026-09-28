"""Build a frozen, non-generative Wikidata English-label snapshot."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import re
import time
from typing import Iterable, Mapping, Sequence

import requests


WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "HyperRAG-research/1.0 (academic reproducibility)"
WIKIDATA_ID_PATTERN = re.compile(r"^[PQ][0-9]+$")


def _load_mapping(path: Path) -> Mapping[str, object]:
    with path.open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - public benchmark format
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def relation_base_id(relation_id: str) -> str:
    """Strip the WikiTopics inverse suffix from a Wikidata property ID."""

    return relation_id[:-4] if relation_id.endswith("_inv") else relation_id


def collect_wikidata_ids(
    structured_root: Path,
    domains: Sequence[str],
) -> tuple[str, ...]:
    """Collect every entity and base relation ID used by the requested domains."""

    identifiers: set[str] = set()
    for domain in domains:
        mapping_path = structured_root / domain / "og_mappings.pkl"
        mappings = _load_mapping(mapping_path)
        for key in ("e2id_train", "e2id_test"):
            values = mappings.get(key)
            if not isinstance(values, Mapping):
                raise ValueError(f"{mapping_path} is missing mapping {key}")
            identifiers.update(str(value) for value in values)
        relations = mappings.get("r2id")
        if not isinstance(relations, Mapping):
            raise ValueError(f"{mapping_path} is missing mapping r2id")
        identifiers.update(relation_base_id(str(value)) for value in relations)
    return tuple(sorted(identifiers))


def _chunks(values: Sequence[str], size: int) -> Iterable[tuple[str, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


def _query_for_labels(identifiers: Sequence[str]) -> str:
    values = " ".join(f"wd:{identifier}" for identifier in identifiers)
    return (
        "SELECT ?item ?itemLabel WHERE { "
        f"VALUES ?item {{ {values} }} "
        'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } '
        "}"
    )


def fetch_label_batch(
    identifiers: Sequence[str],
    *,
    timeout_seconds: float,
    attempts: int = 3,
) -> dict[str, str]:
    """Fetch one label batch and retain only genuine English labels."""

    valid_identifiers = tuple(
        identifier
        for identifier in identifiers
        if WIKIDATA_ID_PATTERN.fullmatch(identifier)
    )
    if not valid_identifiers:
        return {}
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                WIKIDATA_ENDPOINT,
                data={"query": _query_for_labels(valid_identifiers), "format": "json"},
                headers={"User-Agent": USER_AGENT},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            labels: dict[str, str] = {}
            for binding in payload["results"]["bindings"]:
                label = binding.get("itemLabel", {})
                if label.get("xml:lang") != "en":
                    continue
                item_uri = binding["item"]["value"]
                labels[item_uri.rsplit("/", 1)[-1]] = str(label["value"])
            return labels
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Wikidata label batch failed: {identifiers[:3]}") from last_error


def build_label_snapshot(
    structured_root: Path,
    domains: Sequence[str],
    *,
    batch_size: int = 5_000,
    workers: int = 1,
    timeout_seconds: float = 90.0,
) -> dict[str, object]:
    """Fetch all labels needed to replay the released NLG filtering."""

    if batch_size <= 0 or workers <= 0:
        raise ValueError("batch_size and workers must be positive")
    identifiers = collect_wikidata_ids(structured_root, domains)
    batches = tuple(_chunks(identifiers, batch_size))
    labels: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch_label_batch,
                batch,
                timeout_seconds=timeout_seconds,
            ): batch
            for batch in batches
        }
        completed = 0
        for future in as_completed(futures):
            batch_labels = future.result()
            overlap = labels.keys() & batch_labels.keys()
            if overlap:
                raise RuntimeError(f"Duplicate Wikidata IDs across batches: {overlap}")
            labels.update(batch_labels)
            completed += 1
            print(f"label batches: {completed}/{len(batches)}", flush=True)
    return {
        "schema_version": 1,
        "source": WIKIDATA_ENDPOINT,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "domains": list(domains),
        "requested_id_count": len(identifiers),
        "english_label_count": len(labels),
        "missing_english_label_count": len(identifiers) - len(labels),
        "labels": dict(sorted(labels.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    snapshot = build_label_snapshot(
        args.structured_root,
        args.domains,
        batch_size=args.batch_size,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"label snapshot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
