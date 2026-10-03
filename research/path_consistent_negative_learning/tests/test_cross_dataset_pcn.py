"""Structural membership, four strategies and indexed training equivalence."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from research.path_consistent_negative_learning.datasets.ordinary import (
    KnowledgeGraph, Query, DatasetAdapter, training_structure, sample_training_transitions,
    beam_candidates,
)
from research.path_consistent_negative_learning.retriever_only.prepared import method_supervision
from research.path_consistent_negative_learning.retriever_only.training import train_retriever, TrainingConfig
from research.path_consistent_negative_learning.tests.test_retriever_only_training import _training_fixture
from research.path_consistent_negative_learning.scripts import run_experiment


def graph(triples):
    texts = {n: n for t in triples for n in t}
    return KnowledgeGraph.from_triples(triples, texts)


def test_unique_shortest_path_has_no_false_pcn():
    g = graph([("s", "r", "x"), ("x", "r", "a"),
               ("s", "r", "z"), ("z", "r", "w"), ("w", "r", "a")])
    q = Query("q", "q", "s", ("a",))
    structure = training_structure(g, q)
    alternative = ("s", "r", "z")
    modified = (structure[0], (alternative,), structure[2], structure[3])
    assert sample_training_transitions(modified, seed=42, query_key="q")[2] == (False,)


def test_second_complete_shortest_path_is_pcn_but_detour_is_not():
    g = graph([("s", "r", "x"), ("x", "r", "a"),
               ("s", "r", "y"), ("y", "r", "a"),
               ("s", "r", "z"), ("z", "r", "w"), ("w", "r", "a")])
    q = Query("q", "q", "s", ("a",))
    structure = training_structure(g, q)
    second, detour = ("s", "r", "y"), ("w", "r", "a")
    # The detour decreases distance to a, but its prefix already costs two.
    adapter = DatasetAdapter("toy", g, {"train": [q], "valid": [], "test": []}, {})
    assert adapter.identify_path_consistent_negatives(q, (second, detour)) == (True, False)
    assert second not in structure[0]
    union = training_structure(g, q, all_shortest=True)[0]
    assert second in union and ("y", "r", "a") in union and detour not in union


def test_four_strategies_keep_candidates_and_split_and_match_random_count():
    data, _ = _training_fixture()
    original = data.labels.clone()
    labels, weights = method_supervision(data, method="ours", seed=42)
    assert torch.equal(labels, original)
    assert torch.equal(weights == 0, data.path_consistent_mask)
    relabeled, weights = method_supervision(data, method="positive_relabel", seed=42)
    assert torch.equal(relabeled != original, data.path_consistent_mask)
    assert weights.eq(1).all()
    _, random_weights = method_supervision(data, method="matched_random", seed=42)
    for a, b in zip(data.query_offsets[:-1], data.query_offsets[1:]):
        assert int((random_weights[a:b] == 0).sum()) == int(data.path_consistent_mask[a:b].sum())
    assert torch.equal(data.labels, original)


def test_indexed_features_preserve_checkpoint_on_cpu(tmp_path: Path):
    data, embeddings = _training_fixture()
    for method in ("ours", "positive_relabel"):
        checkpoints = []
        for storage in ("indexed", "materialized"):
            p = tmp_path / f"{method}_{storage}.pt"
            train_retriever(data, embeddings, method=method, lambda_=0, seed=42,
                            device_name="cpu", checkpoint_path=p,
                            config=TrainingConfig(maximum_epochs=1, feature_storage=storage))
            checkpoints.append(torch.load(p, weights_only=False)["model_state_dict"])
        assert all(torch.equal(checkpoints[0][k], checkpoints[1][k]) for k in checkpoints[0])


def test_eval_answers_cannot_change_candidates_features_or_training(tmp_path, monkeypatch):
    data, embeddings = _training_fixture()
    g = graph([("n0", "n1", "n2"), ("n0", "n1", "n3")])
    # The fixture embedding store already names these four nodes; remove the
    # inverse edges, which are irrelevant to this one-hop retrieval test.
    g.outgoing = {"n0": (("n1", "n2"), ("n1", "n3"))}
    g.node_texts = {n: n for n in embeddings.node_names}
    first = Query("eval", "question zero", "n0", ("n2",))
    adapter = DatasetAdapter("toy", g, {"train": [], "valid": [], "test": [first]}, {})
    p = tmp_path / "embeddings.pt"
    embeddings.save(p)
    monkeypatch.setattr(run_experiment, "load_dataset", lambda *args: adapter)
    outputs = []
    for i, answers in enumerate((("n2",), ("outside_the_graph",), ())):
        adapter.splits["test"] = [replace(first, answers=answers)]
        root = tmp_path / str(i)
        run_experiment.prepare(SimpleNamespace(dataset="toy", data_root=tmp_path,
                              embeddings=p, device="cpu", output_dir=root, split="test"))
        outputs.append(run_experiment.PreparedCandidates.load(root / "test.pt"))
    for other in outputs[1:]:
        assert other.query_keys == outputs[0].query_keys
        assert other.transitions == outputs[0].transitions
        assert torch.equal(other.dde_features, outputs[0].dde_features)
        assert torch.equal(other.query_embedding_indices, outputs[0].query_embedding_indices)
        assert other.path_consistent_mask is None
    assert torch.equal(data.labels, _training_fixture()[0].labels)
