"""Narrow imports of the released HyperRetriever components."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from HyperRetriever.retrieve.model.dde import DDEEncoder
from HyperRetriever.retrieve.model.mlp import MLP

from .graph import Transition


class OfficialDDE:
    def __init__(self, *, device: str, maximum_hops: int = 3) -> None:
        self.encoder = DDEEncoder(max_hops=maximum_hops, device=device)

    def encode(
        self,
        transitions: Sequence[Transition],
        topic: str,
    ) -> torch.Tensor:
        if not transitions:
            return torch.empty((0, 30), dtype=torch.float32)
        values = self.encoder.compute_dde(list(transitions), [topic])
        features = np.stack([values[transition] for transition in transitions])
        output = torch.from_numpy(features).float()
        if output.shape != (len(transitions), 30):
            raise ValueError(f"official DDE returned shape {tuple(output.shape)}")
        return output


def create_official_mlp(*, device: torch.device) -> MLP:
    return MLP(pred_in_size=4126, emb_size=256).to(device)
