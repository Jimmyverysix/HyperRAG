"""Scheduling contract: all-shortest seeds may overlap only after preparation."""
from pathlib import Path

from research.path_consistent_negative_learning.scripts.build_cross_dataset_manifest import build


def test_all_shortest_has_independent_seeds_and_complete_preparation_dependencies():
    manifest = build("all-shortest", Path("/experiments"))
    jobs = {j["job_id"]: j for j in manifest["jobs"]}
    train = [j for j in jobs.values() if j["commands"][0][3] == "train"]
    assert len(train) == 10
    assert len({j["output_dir"] for j in train}) == 10
    for item in train:
        assert len(item["commands"]) == 1  # no serial five-seed chain
        command = item["commands"][0]
        preparation = jobs[item["depends_on"][0]]
        path = command[command.index("--prepared") + 1]
        assert path in preparation["expected_files"]
        assert "--all-shortest" in preparation["commands"][0]
        assert len(preparation["expected_files"]) == 5
        assert command[command.index("--source-protocol") + 1] == "all_shortest_positive_union_new_negative_pool"
