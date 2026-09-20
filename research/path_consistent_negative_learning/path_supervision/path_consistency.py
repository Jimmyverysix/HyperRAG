"""与训练解耦的完整主题—答案最短路一致性判定。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import networkx as nx

from ..distances import (
    PairDistances,
    build_pair_distance_index,
    on_any_topic_answer_shortest_path,
)
from ..schema import Node, Transition


@dataclass(frozen=True)
class DistanceIndex:
    """固定图与所有有效主题—答案对的距离索引。"""

    graph: nx.Graph
    pairs: tuple[PairDistances, ...]


def build_distance_index(
    graph: nx.Graph,
    source_entities: Iterable[Node],
    answer_entities: Iterable[Node],
) -> DistanceIndex:
    """为 ``S_q × A_q`` 构造可复用距离索引。"""

    sources = tuple(dict.fromkeys(source_entities))
    answers = tuple(dict.fromkeys(answer_entities))
    return DistanceIndex(
        graph=graph,
        pairs=build_pair_distance_index(graph, product(sources, answers)),
    )


def is_path_consistent(
    transition: Transition,
    distance_index: DistanceIndex,
) -> bool:
    """判定是否存在 ``(s,a)`` 满足 ``d(s,v)+2+d(u,a)=d(s,a)``。"""

    return on_any_topic_answer_shortest_path(
        distance_index.graph,
        transition,
        distance_index.pairs,
    )


def find_path_consistent_negatives(
    sampled_negatives: Iterable[Transition],
    distance_index: DistanceIndex,
) -> tuple[bool, ...]:
    """按输入顺序返回路径一致负例的布尔掩码。"""

    return tuple(
        is_path_consistent(transition, distance_index)
        for transition in sampled_negatives
    )
