"""正式的路径弱监督实现。

该包只包含方法定义与可复用核心逻辑；历史协议、训练编排和报告代码保留在外层。
"""

from .path_consistency import (
    DistanceIndex,
    build_distance_index,
    find_path_consistent_negatives,
    is_path_consistent,
)
from .weighted_loss import (
    path_consistent_weak_negative_loss,
    weighted_binary_cross_entropy,
)

__all__ = [
    "DistanceIndex",
    "build_distance_index",
    "find_path_consistent_negatives",
    "is_path_consistent",
    "path_consistent_weak_negative_loss",
    "weighted_binary_cross_entropy",
]
