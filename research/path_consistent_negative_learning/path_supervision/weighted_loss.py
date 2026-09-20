"""路径一致弱负例的加权二元交叉熵。"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def weighted_binary_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """计算 ``sum(w_i BCE_i) / sum(w_i)``。

    归一化只使用当前输入中的有效监督权重，因此 ``lambda=0`` 与删除对应
    样本完全等价，``lambda=1`` 与普通 mean BCE 完全等价。
    """

    if logits.shape != labels.shape or logits.shape != weights.shape:
        raise ValueError("logits、labels 与 weights 必须具有相同形状")
    if not logits.is_floating_point() or not labels.is_floating_point():
        raise TypeError("logits 与 labels 必须是浮点张量")
    if not weights.is_floating_point():
        raise TypeError("weights 必须是浮点张量")
    if torch.any(weights < 0):
        raise ValueError("损失权重不能为负")

    denominator = weights.sum()
    if denominator.detach().item() <= 0.0:
        raise ValueError("损失权重之和必须大于零")
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
    )
    return (elementwise * weights).sum() / denominator


def path_consistent_weak_negative_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    path_consistent_negative_mask: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """仅降低路径一致负例的监督权重，不改变二元标签。"""

    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda 必须位于 [0, 1]")
    if path_consistent_negative_mask.shape != labels.shape:
        raise ValueError("路径一致掩码必须与 labels 具有相同形状")
    if path_consistent_negative_mask.dtype is not torch.bool:
        raise TypeError("路径一致掩码必须是布尔张量")
    if torch.any(path_consistent_negative_mask & (labels != 0)):
        raise ValueError("只能对负例降低监督权重")

    weights = torch.ones_like(logits)
    weights.masked_fill_(path_consistent_negative_mask, lambda_)
    return weighted_binary_cross_entropy(logits, labels, weights)
