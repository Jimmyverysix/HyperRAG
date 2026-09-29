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

    def encode_many(
        self,
        groups: Sequence[tuple[Sequence[Transition], str]],
    ) -> list[torch.Tensor]:
        """Encode independent candidate graphs in one official DDE call.

        Entity and hyperedge IDs are namespaced per group, so the combined
        graph is a disjoint union.  The official propagation, edge order, and
        float32 operations within every group are unchanged; batching only
        removes repeated Python and CUDA-launch overhead.
        """

        combined: list[Transition] = []
        combined_topics: list[str] = []
        namespaced_groups: list[list[Transition]] = []
        for group_index, (transitions, topic) in enumerate(groups):
            prefix = f"{group_index}:"
            values = [
                (prefix + head, prefix + edge, prefix + tail)
                for head, edge, tail in transitions
            ]
            namespaced_groups.append(values)
            combined.extend(values)
            combined_topics.append(prefix + topic)
        if not combined:
            return [
                torch.empty((0, 30), dtype=torch.float32)
                for transitions, _ in groups
            ]
        values = self.encoder.compute_dde(combined, combined_topics)
        outputs = []
        for namespaced in namespaced_groups:
            if not namespaced:
                outputs.append(torch.empty((0, 30), dtype=torch.float32))
                continue
            features = np.stack([values[transition] for transition in namespaced])
            output = torch.from_numpy(features).float()
            if output.shape != (len(namespaced), 30):
                raise ValueError(
                    f"official batched DDE returned shape {tuple(output.shape)}"
                )
            outputs.append(output)
        return outputs


def create_official_mlp(*, device: torch.device) -> MLP:
    return MLP(pred_in_size=4126, emb_size=256).to(device)
