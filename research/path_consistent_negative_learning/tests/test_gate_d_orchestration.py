import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import torch

from research.path_consistent_negative_learning.prepare_gate_d import (
    GateDPreparationTask,
    prepare_gate_d_data,
)
from research.path_consistent_negative_learning.run_gate_d_all import run_all_gate_d
from research.path_consistent_negative_learning.structured_data import (
    StructuredExperimentData,
)


DOMAINS = tuple(f"d{index}" for index in range(11))
SEEDS = (142, 143, 144, 145, 146)


def write_protocol(path: Path) -> None:
    training = {
        "epochs": 50,
        "patience": 8,
        "batch_size": 4096,
        "learning_rate": 0.001,
    }
    payload = {
        "schema_version": 1,
        "status": "frozen_before_weighted_results",
        "selection": {"candidate_weights": [0.25, 0.5]},
        "independent_confirmation": {
            "domain": "award",
            "reported_split": "test",
        },
        "gate_d": {
            "run_only_after_independent_confirmation_passes": True,
            "locked_weight_across_domains": True,
            "domains": list(DOMAINS),
            "sampler_and_training_seeds": list(SEEDS),
            "split_seed": 99,
            "split_scheme": "70/15/15",
            "reported_split": "test",
            "training": training,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_dataset(
    path: Path,
    *,
    domain: str,
    seed: int,
    split_seed: int = 99,
    split_scheme: str = "70/15/15",
) -> None:
    data = StructuredExperimentData(
        domain=domain,
        sampler_seed=seed,
        query_keys=[f"{domain}:0"],
        query_relations=torch.tensor([[0, 0, 0]], dtype=torch.long),
        query_topics=torch.tensor([0], dtype=torch.long),
        query_splits=torch.tensor([2], dtype=torch.uint8),
        query_answer_ids=[(0,)],
        query_offsets=torch.tensor([0, 1], dtype=torch.long),
        candidate_query_indices=torch.tensor([0], dtype=torch.long),
        candidate_transitions=torch.tensor([[0, 0, 0]], dtype=torch.long),
        numeric_features=torch.zeros((1, 9), dtype=torch.float32),
        selected_labels=torch.tensor([True]),
        disputed_labels=torch.tensor([False]),
        hyperedge_relation_mask=torch.ones((1, 1), dtype=torch.float32),
        entity_count=1,
        relation_count=1,
        split_seed=split_seed,
        split_scheme=split_scheme,
    )
    data.save(path)


def write_confirmation(path: Path, *, proceed: bool = True) -> None:
    payload = {
        "experiment": "weighted_strategy2_independent_confirmation",
        "domain": "award",
        "evaluation_split": "test",
        "locked_weight": 0.25,
        "gate": {"proceed_to_gate_d": proceed},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class GateDPreparationTests(unittest.TestCase):
    def test_prepares_frozen_11_by_5_grid_with_at_most_six_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            dataset_root = root / "source"
            output_root = root / "prepared"
            write_protocol(protocol_path)
            for domain in DOMAINS:
                (dataset_root / domain).mkdir(parents=True)

            state = {"active": 0, "maximum": 0}
            lock = threading.Lock()

            def fake_runner(task: GateDPreparationTask) -> None:
                with lock:
                    state["active"] += 1
                    state["maximum"] = max(state["maximum"], state["active"])
                try:
                    time.sleep(0.005)
                    write_dataset(
                        task.output_path,
                        domain=task.domain,
                        seed=task.seed,
                        split_seed=task.split_seed,
                        split_scheme=task.split_scheme,
                    )
                finally:
                    with lock:
                        state["active"] -= 1

            summary = prepare_gate_d_data(
                dataset_root,
                output_root,
                protocol_path=protocol_path,
                max_workers=6,
                task_runner=fake_runner,
            )

            self.assertEqual(summary, {"prepared": 55, "skipped": 0, "total": 55})
            self.assertLessEqual(state["maximum"], 6)
            self.assertGreater(state["maximum"], 1)
            self.assertTrue((output_root / "prepare_manifest.json").is_file())
            self.assertTrue((output_root / "COMPLETE").is_file())

    def test_resume_rejects_existing_dataset_with_wrong_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            dataset_root = root / "source"
            output_root = root / "prepared"
            write_protocol(protocol_path)
            for domain in DOMAINS:
                (dataset_root / domain).mkdir(parents=True)
            write_dataset(
                output_root / DOMAINS[0] / "datasets" / f"seed_{SEEDS[0]}.pt",
                domain="wrong-domain",
                seed=SEEDS[0],
            )

            with self.assertRaisesRegex(ValueError, "冻结协议"):
                prepare_gate_d_data(
                    dataset_root,
                    output_root,
                    protocol_path=protocol_path,
                    task_runner=lambda task: self.fail("不应启动任何新任务"),
                )
            self.assertFalse((output_root / "COMPLETE").exists())


class GateDRunTests(unittest.TestCase):
    def test_runs_domains_sequentially_with_locked_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            data_root = root / "prepared"
            output_root = root / "runs"
            confirmation_path = root / "confirmation_summary.json"
            write_protocol(protocol_path)
            write_confirmation(confirmation_path)
            for domain in DOMAINS:
                for seed in SEEDS:
                    write_dataset(
                        data_root / domain / "datasets" / f"seed_{seed}.pt",
                        domain=domain,
                        seed=seed,
                    )

            calls = []
            active = 0

            def fake_suite(**kwargs) -> None:
                nonlocal active
                active += 1
                self.assertEqual(active, 1)
                try:
                    calls.append(kwargs)
                    kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
                    (kwargs["output_dir"] / "COMPLETE").write_text(
                        "done\n", encoding="utf-8"
                    )
                finally:
                    active -= 1

            summary = run_all_gate_d(
                data_root,
                output_root,
                confirmation_path,
                protocol_path=protocol_path,
                suite_runner=fake_suite,
            )

            self.assertEqual([call["domain"] for call in calls], list(DOMAINS))
            self.assertEqual(summary["domain_count"], 11)
            for call in calls:
                self.assertEqual(call["phase"], "gate_d")
                self.assertEqual(call["gpu_ids"], (0, 1, 2, 3, 4, 5))
                self.assertEqual(call["strategy2_weights"], (0.25,))
                self.assertEqual(call["random_control_weights"], (0.25,))
            self.assertTrue((output_root / "run_manifest.json").is_file())
            self.assertTrue((output_root / "COMPLETE").is_file())

    def test_refuses_gate_d_before_confirmation_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            confirmation_path = root / "confirmation_summary.json"
            write_protocol(protocol_path)
            write_confirmation(confirmation_path, proceed=False)

            with self.assertRaisesRegex(RuntimeError, "禁止启动门槛 D"):
                run_all_gate_d(
                    root / "prepared",
                    root / "runs",
                    confirmation_path,
                    protocol_path=protocol_path,
                    suite_runner=lambda **kwargs: self.fail("不应启动训练"),
                )


if __name__ == "__main__":
    unittest.main()
