"""Checks for the scientific aggregation unit and complete paired arms."""
import json
import numpy as np
import pytest

from research.path_consistent_negative_learning.scripts.aggregate_cross_dataset_results import (
    average_reports, bootstrap, summarize, SEEDS,
)


def test_equal_domain_bootstrap_does_not_weight_large_domain_more():
    small = np.ones((1, 3))
    large = np.zeros((100, 3))
    result = bootstrap({"small": small, "large": large}, resamples=200)
    for value in result.values():
        assert value == {"difference": .5, "ci_low": .5, "ci_high": .5}


def test_missing_seed_has_no_formal_mean_and_paired_coverage_is_required(tmp_path):
    paths = []
    for seed in SEEDS:
        path = tmp_path / f"{seed}.json";paths.append(path)
        if seed != 46:
            path.write_text(json.dumps({"seed": seed, "queries": [{"query_key": "q", "candidate_count": 1,
                "reciprocal_rank": 1, "answer_reach_10": 1, "answer_reach_5": 1}]}))
    assert average_reports(paths) is None
    paths[-1].write_text(json.dumps({"seed": 46, "queries": [{"query_key": "different", "candidate_count": 1,
        "reciprocal_rank": 1, "answer_reach_10": 1, "answer_reach_5": 1}]}))
    with pytest.raises(ValueError, match="coverage"):
        average_reports(paths)
