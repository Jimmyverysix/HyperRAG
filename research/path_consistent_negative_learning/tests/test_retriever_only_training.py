from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from research.path_consistent_negative_learning.retriever_only.embeddings import (
    EmbeddingStore,
)
from research.path_consistent_negative_learning.retriever_only.prepared import (
    PreparedCandidates,
)
from research.path_consistent_negative_learning.retriever_only import training
from research.path_consistent_negative_learning.retriever_only.training import (
    TrainingConfig,
    _index_batches,
    train_retriever,
)


def _legacy_loader(
    indices: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def _legacy_index_batches(
    indices: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
    shuffle: bool,
):
    loader = DataLoader(
        TensorDataset(indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )
    for (batch,) in loader:
        yield batch


def _training_fixture() -> tuple[PreparedCandidates, EmbeddingStore]:
    count = 40
    labels = torch.tensor([(index % 2) == 0 for index in range(count)])
    path_mask = torch.tensor(
        [(index % 4) == 1 for index in range(count)], dtype=torch.bool
    )
    data = PreparedCandidates(
        domain="fixture",
        split="train",
        seed=42,
        query_keys=["q0", "q1"],
        query_topics=["n0", "n1"],
        query_answers=[("n2",), ("n3",)],
        query_offsets=torch.tensor([0, 20, 40]),
        query_embedding_indices=torch.tensor([0] * 20 + [1] * 20),
        head_embedding_indices=torch.arange(count) % 4,
        edge_embedding_indices=(torch.arange(count) + 1) % 4,
        tail_embedding_indices=(torch.arange(count) + 2) % 4,
        dde_features=torch.randn(count, 30, generator=torch.Generator().manual_seed(5)),
        labels=labels,
        path_consistent_mask=path_mask,
    )
    embeddings = EmbeddingStore(
        node_names=("n0", "n1", "n2", "n3"),
        node_embeddings=torch.randn(
            4, 1024, generator=torch.Generator().manual_seed(6)
        ),
        query_texts=("question zero", "question one"),
        query_embeddings=torch.randn(
            2, 1024, generator=torch.Generator().manual_seed(7)
        ),
    )
    data.validate()
    embeddings.validate()
    return data, embeddings


class RetrieverTrainingTests(unittest.TestCase):
    def test_direct_shuffled_batches_match_dataloader_across_epochs(self) -> None:
        indices = torch.tensor([41, 7, 19, 101, 3, 89, 23, 67, 11, 5, 47])
        legacy = _legacy_loader(indices, batch_size=4, seed=43, shuffle=True)
        generator = torch.Generator().manual_seed(43)
        for _ in range(5):
            expected = [batch[0] for batch in legacy]
            actual = list(
                _index_batches(
                    indices,
                    batch_size=4,
                    generator=generator,
                    shuffle=True,
                )
            )
            self.assertEqual(len(actual), len(expected))
            for actual_batch, expected_batch in zip(actual, expected):
                self.assertTrue(torch.equal(actual_batch, expected_batch))

    def test_direct_sequential_batches_match_dataloader(self) -> None:
        indices = torch.tensor([31, 2, 71, 13, 5, 17, 29])
        legacy = _legacy_loader(indices, batch_size=3, seed=44, shuffle=False)
        generator = torch.Generator().manual_seed(44)
        for _ in range(3):
            expected = [batch[0] for batch in legacy]
            actual = list(
                _index_batches(
                    indices,
                    batch_size=3,
                    generator=generator,
                    shuffle=False,
                )
            )
            self.assertEqual(len(actual), len(expected))
            for actual_batch, expected_batch in zip(actual, expected):
                self.assertTrue(torch.equal(actual_batch, expected_batch))

    def test_rejects_nonpositive_batch_size(self) -> None:
        batches = _index_batches(
            torch.arange(3),
            batch_size=0,
            generator=torch.Generator().manual_seed(1),
            shuffle=True,
        )
        with self.assertRaisesRegex(ValueError, "batch_size"):
            next(batches)

    def test_training_history_and_checkpoint_match_legacy_loader(self) -> None:
        data, embeddings = _training_fixture()
        config = TrainingConfig(
            batch_size=4,
            maximum_epochs=4,
            patience=4,
            feature_materialization_chunk_size=16,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_path = root / "legacy.pt"
            direct_path = root / "direct.pt"
            with patch.object(training, "_index_batches", _legacy_index_batches):
                legacy = train_retriever(
                    data,
                    embeddings,
                    method="ours",
                    lambda_=0.25,
                    seed=42,
                    device_name="cpu",
                    checkpoint_path=legacy_path,
                    config=config,
                )
            direct = train_retriever(
                data,
                embeddings,
                method="ours",
                lambda_=0.25,
                seed=42,
                device_name="cpu",
                checkpoint_path=direct_path,
                config=config,
            )

            self.assertEqual(legacy["history"], direct["history"])
            legacy_state = torch.load(
                legacy_path, map_location="cpu", weights_only=False
            )["model_state_dict"]
            direct_state = torch.load(
                direct_path, map_location="cpu", weights_only=False
            )["model_state_dict"]
            self.assertEqual(legacy_state.keys(), direct_state.keys())
            for name in legacy_state:
                self.assertTrue(
                    torch.equal(legacy_state[name], direct_state[name]), name
                )


if __name__ == "__main__":
    unittest.main()
