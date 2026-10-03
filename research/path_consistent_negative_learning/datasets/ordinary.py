"""Ordinary KG transitions for MetaQA, KQA Pro and PathQuestion.

No gold relation sequence or per-question gold subgraph is used for retrieval.
Dataset labels supply topics and answers; the fixed KG supplies all edges.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import json
import random
import re
from typing import Callable, Iterable

from ..path_supervision.dag import on_shortest_answer_dag

Transition = tuple[str, str, str]


@dataclass(frozen=True)
class Query:
    key: str
    text: str
    topic: str
    answers: tuple[str, ...]


@dataclass
class KnowledgeGraph:
    outgoing: dict[str, tuple[tuple[str, str], ...]]
    incoming: dict[str, tuple[tuple[str, str], ...]]
    node_texts: dict[str, str]
    triple_count: int

    @classmethod
    def from_triples(cls, triples: Iterable[Transition], texts: dict[str, str]):
        outgoing, incoming = defaultdict(set), defaultdict(set)
        triples = set(triples)
        for head, edge, tail in triples:
            inverse = edge + "_inv"
            texts.setdefault(inverse, "inverse of " + texts[edge])
            for source, relation, target in ((head, edge, tail), (tail, inverse, head)):
                outgoing[source].add((relation, target))
                incoming[target].add((source, relation))
        return cls({n: tuple(sorted(v)) for n, v in outgoing.items()},
                   {n: tuple(sorted(v)) for n, v in incoming.items()}, texts, len(triples))


@dataclass
class DatasetAdapter:
    name: str
    graph: KnowledgeGraph
    splits: dict[str, list[Query]]
    metadata: dict
    maximum_hops: int = 3

    def load_train(self):
        return self.splits["train"]

    def load_validation(self):
        return self.splits["valid"]

    def load_test(self):
        return self.splits["test"]

    def get_question(self, query):
        return query.text

    def get_topic_entities(self, query):
        return (query.topic,)

    def get_answer_entities(self, query):
        return query.answers

    def build_graph(self):
        return self.graph

    def shortest_distance(self, topic):
        return shortest_tree(self.graph, topic, self.maximum_hops)[0]

    def get_candidate_transitions(self, topic, score, width=32):
        return beam_candidates(self.graph, topic, score, width=width,
                               maximum_hops=self.maximum_hops)

    def identify_path_consistent_negatives(self, query, negatives):
        distances, _ = shortest_tree(self.graph, query.topic, self.maximum_hops)
        nodes = answer_dag_nodes(self.graph, distances, query.answers)
        return tuple(on_shortest_answer_dag(h, t, distances, nodes, transition_cost=1)
                     for h, _, t in negatives)


def shortest_tree(graph, topic, maximum_hops=3):
    distances, parents = {topic: 0}, {topic: None}
    queue = deque([topic])
    while queue:
        head = queue.popleft()
        if distances[head] >= maximum_hops:
            continue
        for edge, tail in graph.outgoing.get(head, ()):
            if tail not in distances:
                distances[tail], parents[tail] = distances[head] + 1, (head, edge)
                queue.append(tail)
    return distances, parents


def answer_dag_nodes(graph, distances, answers):
    nodes = {a for a in answers if a in distances}
    queue = deque(nodes)
    while queue:
        tail = queue.popleft()
        for head, _ in graph.incoming.get(tail, ()):
            if distances.get(head) == distances[tail] - 1 and head not in nodes:
                nodes.add(head)
                queue.append(head)
    return nodes


def training_structure(graph, query, *, topic_cache=None, maximum_hops=3,
                       all_shortest=False):
    if topic_cache is None:
        distances, parents = shortest_tree(graph, query.topic, maximum_hops)
    else:
        if query.topic not in topic_cache:
            topic_cache[query.topic] = shortest_tree(graph, query.topic, maximum_hops)
        distances, parents = topic_cache[query.topic]
    answers = tuple(a for a in query.answers if 1 <= distances.get(a, maximum_hops + 1) <= maximum_hops)
    if not answers:
        return None
    positives, selected_nodes = set(), {query.topic}
    for answer in answers:
        node = answer
        while node != query.topic:
            head, edge = parents[node]
            positives.add((head, edge, node))
            selected_nodes.add(node)
            node = head
    on_path = answer_dag_nodes(graph, distances, answers)
    if all_shortest:
        positives = {(h, e, t) for h in on_path for e, t in graph.outgoing.get(h, ())
                     if on_shortest_answer_dag(h, t, distances, on_path, transition_cost=1)}
        selected_nodes = on_path
    pool, frontier = set(), {query.topic}
    for _ in range(max(distances[a] for a in answers)):
        following = set()
        for head in frontier:
            for edge, tail in graph.outgoing.get(head, ()):
                pool.add((head, edge, tail))
                if tail in selected_nodes:
                    following.add(tail)
        frontier = following
    return tuple(sorted(positives)), tuple(sorted(pool - positives)), distances, on_path


def sample_training_transitions(structure, *, seed, query_key):
    positives, pool, distances, nodes = structure
    negatives = tuple(sorted(random.Random(f"{seed}|{query_key}").sample(
        pool, min(len(positives), len(pool)))))
    mask = tuple(on_shortest_answer_dag(h, t, distances, nodes, transition_cost=1)
                 for h, _, t in negatives)
    return positives, negatives, mask


def beam_candidates(graph, topic, score: Callable[[str], float], *, width=32,
                    maximum_hops=3):
    frontier, expanded, selected, output = {topic}, set(), set(), []
    for _ in range(maximum_hops):
        choices = []
        for head in sorted(frontier):
            if head in expanded:
                continue
            for edge, tail in graph.outgoing.get(head, ()):
                transition = head, edge, tail
                if transition not in selected:
                    choices.append(((score(head) + score(edge) + score(tail)) / 3, transition))
        chosen = sorted(choices, key=lambda x: (-x[0], x[1]))[:width]
        if not chosen:
            break
        frontier = set()
        for _, transition in chosen:
            selected.add(transition)
            output.append(transition)
            frontier.add(transition[2])
        expanded.update(t[0] for _, t in chosen)
    return tuple(output)


def load_metaqa(root: Path):
    texts, triples = {}, []
    for line in (root / "kb.txt").read_text(encoding="utf-8").splitlines():
        h, r, t = line.split("|", 2)
        texts.update({"E:" + h: h, "E:" + t: t, "R:" + r: r.replace("_", " ")})
        triples.append(("E:" + h, "R:" + r, "E:" + t))
    splits = {}
    for split, filename in (("train", "qa_train.txt"), ("valid", "qa_dev.txt"), ("test", "qa_test.txt")):
        rows = []
        for i, line in enumerate((root / filename).read_text(encoding="utf-8").splitlines()):
            question, answer = line.split("\t", 1)
            topic = re.search(r"\[([^\]]+)\]", question).group(1)
            # Keep the historical query keys to reuse frozen sampled stores.
            rows.append(Query(f"metaqa_3hop:{'dev' if split == 'valid' else split}:{i}",
                              question, "E:" + topic,
                              tuple(sorted({"E:" + a for a in answer.split("|")}))))
        splits[split] = rows
    return DatasetAdapter("MetaQA-3hop-vanilla", KnowledgeGraph.from_triples(triples, texts),
                          splits, {"version": "official_3hop_vanilla", "graph_projection": "relation_edges_with_inverse"})


def load_pathquestion(root: Path, *, split_seed=20261003):
    texts, triples = {}, set()
    rows = {}
    for hop in (2, 3):
        for line in (root / f"{hop}H-kb.txt").read_text(encoding="utf-8").splitlines():
            h, r, t = line.split("\t")
            texts.update({"E:" + h: h.replace("_", " "), "E:" + t: t.replace("_", " "),
                          "R:" + r: r.replace("_", " ")})
            triples.add(("E:" + h, "R:" + r, "E:" + t))
        for line in (root / f"{hop}H.txt").read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            text, path = fields[0], fields[2]
            topic = path.split("#", 1)[0]
            answers = tuple(sorted({"E:" + a for a in fields[3].split("/") if a}))
            # Original files contain paraphrases and duplicated questions.
            # Group by the exact question text before splitting, keeping all
            # its labels together. This prevents duplicate-text leakage.
            key = (text, topic)
            if key in rows:
                old = rows[key]
                answers = tuple(sorted(set(answers) | set(old.answers)))
            rows[key] = Query("", text, "E:" + topic, answers)
    ordered = sorted(rows.values(), key=lambda q: (q.text, q.topic))
    random.Random(split_seed).shuffle(ordered)
    n = len(ordered)
    boundaries = (0, int(n * .8), int(n * .9), n)
    splits = {}
    for split, start, stop in zip(("train", "valid", "test"), boundaries, boundaries[1:]):
        splits[split] = [Query(f"PathQuestion:{split}:{i}", q.text, q.topic, q.answers)
                         for i, q in enumerate(ordered[start:stop])]
    return DatasetAdapter("PathQuestion-2H3H", KnowledgeGraph.from_triples(triples, texts), splits,
                          {"version": "zmtkeke_IRN_PQ_2H3H", "split_seed": split_seed,
                           "split_protocol": "exact_question_topic_grouped_80_10_10",
                           "raw_files": ["2H.txt", "3H.txt"],
                           "uses_gold_path_for": "topic_only; not graph, candidates, features or selected positive path"})


def load_kqapro(root: Path, *, split_seed=20261003):
    kb = json.loads((root / "kb.json").read_text(encoding="utf-8"))
    entities = kb["entities"]
    texts = {"E:" + k: v["name"] for k, v in entities.items()}
    by_name = defaultdict(list)
    for k, v in entities.items():
        by_name[v["name"]].append("E:" + k)
    triples = set()
    for head, item in entities.items():
        for edge in item.get("relations", []):
            tail = edge["object"]
            if tail not in entities:
                continue
            relation = "R:" + edge["predicate"]
            texts[relation] = edge["predicate"]
            h, t = "E:" + head, "E:" + tail
            if edge.get("direction", "forward") == "backward":
                h, t = t, h
            triples.add((h, relation, t))
    raw_counts, rejected, parsed = {}, {}, {}
    for source in ("train", "val"):
        raw = json.loads((root / f"{source}.json").read_text(encoding="utf-8"))
        raw_counts[source] = len(raw)
        rows, reasons = [], defaultdict(int)
        for i, item in enumerate(raw):
            program = item["program"]
            finds = [step for step in program if step["function"] == "Find"]
            if not program or program[-1]["function"] != "What":
                reasons["non_entity_output"] += 1
                continue
            if len(finds) != 1 or len(by_name.get(finds[0]["inputs"][0], [])) != 1:
                reasons["not_one_unambiguous_topic"] += 1
                continue
            topic = by_name[finds[0]["inputs"][0]][0]
            rows.append(Query(f"KQAPro:{source}:{i}", item["question"], topic,
                              tuple(sorted(by_name.get(item.get("answer", ""), [])))))
        rejected[source] = dict(reasons)
        parsed[source] = rows
    # Official test has no public gold answers. Use official validation as a
    # held-out evaluation split, and reserve 10% of training for development.
    train = sorted(parsed["train"], key=lambda q: q.key)
    random.Random(split_seed).shuffle(train)
    cutoff = int(len(train) * .9)
    return DatasetAdapter("KQAPro-entity", KnowledgeGraph.from_triples(triples, texts),
                          {"train": train[:cutoff], "valid": train[cutoff:], "test": parsed["val"]},
                          {"version": "official_KQAPro_mirror_drt", "split_seed": split_seed,
                           "raw_counts": raw_counts, "exclusion_counts": rejected,
                           "evaluation_split": "official_val_entity_output_single_topic_subset",
                           "development_split": "seeded_10pct_of_official_train_subset",
                           "graph_projection": "entity_relation_only; attributes, qualifiers and concepts excluded",
                           "topic_source": "sole unambiguous Find annotation",
                           "task_subset": "terminal What and one unambiguous Find; no answer_reachability_filter"})


def load_dataset(name: str, root: Path):
    return {"metaqa": load_metaqa, "pathquestion": load_pathquestion,
            "kqapro": load_kqapro}[name](root)
