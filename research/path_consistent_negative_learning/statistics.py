"""Strictly paired statistical summaries for retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, TypeAlias

import numpy as np
from numpy.typing import NDArray


KeyedValues: TypeAlias = Mapping[Hashable, float] | Iterable[tuple[Hashable, float]]
SeedValues: TypeAlias = (
    Mapping[Hashable, KeyedValues]
    | Iterable[tuple[Hashable, Hashable, float]]
)


class PairingError(ValueError):
    """Base error for invalid paired observations."""


class DuplicateKeyError(PairingError):
    """Raised when one side contains a repeated key."""


class MissingKeyError(PairingError):
    """Raised when the two sides do not contain exactly the same keys."""


@dataclass(frozen=True)
class PairedValues:
    """Validated arrays aligned by the reference side's key order."""

    keys: tuple[Hashable, ...]
    reference: NDArray[np.float64]
    comparison: NDArray[np.float64]


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Percentile paired-bootstrap CI for ``comparison - reference``."""

    n_pairs: int
    reference_mean: float
    comparison_mean: float
    mean_difference: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_resamples: int
    seed: int


@dataclass(frozen=True)
class SeedDifference:
    """Paired mean difference for one training seed."""

    seed: Hashable
    n_queries: int
    reference_mean: float
    comparison_mean: float
    mean_difference: float


@dataclass(frozen=True)
class SeedPairedSummary:
    """Equal-seed-weighted summary of query-paired experiment results."""

    seed_count: int
    per_seed: dict[Hashable, SeedDifference]
    reference_mean: float
    comparison_mean: float
    mean_difference: float
    difference_std: float


def _finite_float(value: object, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PairingError(f"{context} must be a finite numeric value") from exc
    if not np.isfinite(result):
        raise PairingError(f"{context} must be a finite numeric value")
    return result


def _keyed_dict(values: KeyedValues, *, side: str) -> dict[Hashable, float]:
    items = values.items() if isinstance(values, Mapping) else values
    result: dict[Hashable, float] = {}
    for item in items:
        try:
            key, value = item
        except (TypeError, ValueError) as exc:
            raise PairingError(f"{side} observations must be (key, value) pairs") from exc
        try:
            is_duplicate = key in result
        except TypeError as exc:
            raise PairingError(f"{side} key {key!r} is not hashable") from exc
        if is_duplicate:
            raise DuplicateKeyError(f"duplicate key on {side} side: {key!r}")
        result[key] = _finite_float(value, context=f"value for {side} key {key!r}")
    return result


def _format_keys(keys: set[Hashable]) -> str:
    return ", ".join(sorted((repr(key) for key in keys)))


def pair_by_key(reference: KeyedValues, comparison: KeyedValues) -> PairedValues:
    """Validate and align two value collections by their exact query keys.

    Iterable inputs are accepted so duplicate keys can be detected.  Missing or
    extra keys are errors; observations are never paired by position.
    """

    reference_values = _keyed_dict(reference, side="reference")
    comparison_values = _keyed_dict(comparison, side="comparison")
    reference_keys = set(reference_values)
    comparison_keys = set(comparison_values)
    missing_from_comparison = reference_keys - comparison_keys
    missing_from_reference = comparison_keys - reference_keys
    if missing_from_comparison or missing_from_reference:
        details: list[str] = []
        if missing_from_comparison:
            details.append(
                "missing from comparison: " + _format_keys(missing_from_comparison)
            )
        if missing_from_reference:
            details.append(
                "missing from reference: " + _format_keys(missing_from_reference)
            )
        raise MissingKeyError("; ".join(details))
    if not reference_values:
        raise PairingError("at least one paired observation is required")

    keys = tuple(reference_values)
    return PairedValues(
        keys=keys,
        reference=np.asarray([reference_values[key] for key in keys], dtype=np.float64),
        comparison=np.asarray([comparison_values[key] for key in keys], dtype=np.float64),
    )


def paired_bootstrap_mean_difference(
    reference: KeyedValues,
    comparison: KeyedValues,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Estimate a paired-bootstrap CI for the mean difference.

    Query keys are checked before sampling.  Each bootstrap draw resamples
    paired query differences, so the result is ``comparison - reference``.
    ``seed`` defaults to zero to make the analysis exactly reproducible.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    paired = pair_by_key(reference, comparison)
    differences = paired.comparison - paired.reference
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_resamples, dtype=np.float64)

    # Bound peak memory while retaining the exact RNG stream for a given seed.
    batch_size = max(1, min(n_resamples, 1_000_000 // len(differences)))
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        indices = rng.integers(
            0,
            len(differences),
            size=(stop - start, len(differences)),
        )
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)

    tail_probability = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [tail_probability, 1.0 - tail_probability],
    )
    return PairedBootstrapResult(
        n_pairs=len(differences),
        reference_mean=float(paired.reference.mean()),
        comparison_mean=float(paired.comparison.mean()),
        mean_difference=float(differences.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
    )


def _seed_dict(values: SeedValues, *, side: str) -> dict[Hashable, dict[Hashable, float]]:
    if isinstance(values, Mapping):
        result: dict[Hashable, dict[Hashable, float]] = {}
        for seed, observations in values.items():
            try:
                duplicate_seed = seed in result
            except TypeError as exc:
                raise PairingError(f"{side} seed {seed!r} is not hashable") from exc
            if duplicate_seed:
                raise DuplicateKeyError(f"duplicate seed on {side} side: {seed!r}")
            result[seed] = _keyed_dict(
                observations,
                side=f"{side}, seed {seed!r}",
            )
        return result

    result = {}
    for item in values:
        try:
            seed, query_key, value = item
        except (TypeError, ValueError) as exc:
            raise PairingError(
                f"{side} observations must be (seed, query_key, value) triples"
            ) from exc
        try:
            seed_values = result.setdefault(seed, {})
            duplicate_key = query_key in seed_values
        except TypeError as exc:
            raise PairingError(
                f"{side} seed and query keys must be hashable: "
                f"{seed!r}, {query_key!r}"
            ) from exc
        if duplicate_key:
            raise DuplicateKeyError(
                f"duplicate key on {side} side for seed {seed!r}: {query_key!r}"
            )
        seed_values[query_key] = _finite_float(
            value,
            context=f"value for {side} seed {seed!r}, key {query_key!r}",
        )
    return result


def summarize_paired_seeds(
    reference: SeedValues,
    comparison: SeedValues,
) -> SeedPairedSummary:
    """Summarize paired query metrics separately for each random seed.

    Both seed sets and query-key sets within each seed must match exactly.
    Overall means give every seed equal weight; ``difference_std`` is the
    sample standard deviation across per-seed paired mean differences (zero
    when only one seed is supplied).
    """

    reference_seeds = _seed_dict(reference, side="reference")
    comparison_seeds = _seed_dict(comparison, side="comparison")
    missing_comparison = set(reference_seeds) - set(comparison_seeds)
    missing_reference = set(comparison_seeds) - set(reference_seeds)
    if missing_comparison or missing_reference:
        details: list[str] = []
        if missing_comparison:
            details.append(
                "seeds missing from comparison: " + _format_keys(missing_comparison)
            )
        if missing_reference:
            details.append(
                "seeds missing from reference: " + _format_keys(missing_reference)
            )
        raise MissingKeyError("; ".join(details))
    if not reference_seeds:
        raise PairingError("at least one seed is required")

    per_seed: dict[Hashable, SeedDifference] = {}
    for seed in reference_seeds:
        try:
            paired = pair_by_key(reference_seeds[seed], comparison_seeds[seed])
        except PairingError as exc:
            raise type(exc)(f"seed {seed!r}: {exc}") from exc
        differences = paired.comparison - paired.reference
        per_seed[seed] = SeedDifference(
            seed=seed,
            n_queries=len(differences),
            reference_mean=float(paired.reference.mean()),
            comparison_mean=float(paired.comparison.mean()),
            mean_difference=float(differences.mean()),
        )

    seed_results = list(per_seed.values())
    difference_values = np.asarray(
        [result.mean_difference for result in seed_results],
        dtype=np.float64,
    )
    return SeedPairedSummary(
        seed_count=len(seed_results),
        per_seed=per_seed,
        reference_mean=float(np.mean([result.reference_mean for result in seed_results])),
        comparison_mean=float(np.mean([result.comparison_mean for result in seed_results])),
        mean_difference=float(difference_values.mean()),
        difference_std=(
            float(difference_values.std(ddof=1)) if len(difference_values) > 1 else 0.0
        ),
    )
