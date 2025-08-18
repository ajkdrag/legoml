from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple, Dict
import torch
import torch.nn.functional as F

from legoml.core.contracts.task import Task, Scalars
from legoml.core.contracts.objective import Objective
from legoml.core.contracts.embedder import TextEmbedder, ImageEmbedder
from legoml.core.contracts.metric import Metric
from legoml.data.batches import PairBatch


def default_pair_aggregator(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    # concat of z1 and z2; users can replace with [z1, z2, |z1-z2|, z1*z2]
    return torch.cat([z1, z2], dim=-1)


@dataclass
class PairSimilarityTask(Task):
    text_embedder: TextEmbedder
    image_embedder: ImageEmbedder
    classifier: torch.nn.Module  # expects pair feature vector as input
    metrics: List[Metric]
    device: str
    aggregator: Any = default_pair_aggregator

    def on_epoch_start(self, *, mode: str, epoch: int) -> None:
        for m in self.metrics:
            m.reset()

    def on_epoch_end(self, *, mode: str, epoch: int) -> Scalars:
        scalars: Scalars = {}
        for m in self.metrics:
            scalars.update(m.compute())
        return {f"{mode}/{k}": float(v) for k, v in scalars.items()}

    @property
    def model(self) -> torch.nn.Module:
        return self.classifier

    @model.setter
    def model(self, value: torch.nn.Module) -> None:
        self.classifier = value

    def _forward(self, batch: PairBatch) -> tuple[torch.Tensor, torch.Tensor]:
        t1 = self.text_embedder.encode(batch.text1).to(self.device)
        i1 = self.image_embedder.encode(batch.image1).to(self.device)
        z1 = torch.cat([t1, i1], dim=-1)

        t2 = self.text_embedder.encode(batch.text2).to(self.device)
        i2 = self.image_embedder.encode(batch.image2).to(self.device)
        z2 = torch.cat([t2, i2], dim=-1)
        pair_feat = self.aggregator(z1, z2)
        logits = self.classifier(pair_feat).squeeze(-1)
        return logits, batch.labels.to(self.device)

    def train_step(
        self, batch: PairBatch, *, step: int, epoch: int
    ) -> tuple[torch.Tensor, Scalars]:
        logits, targets = self._forward(batch)
        loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        # update metrics expecting outputs, targets (use logits convention like other tasks)
        for m in self.metrics:
            m.update(
                outputs=logits.detach().unsqueeze(-1),
                targets=targets.detach().unsqueeze(-1),
            )
        return loss, {"loss/bce": float(loss.detach().item())}

    def eval_step(
        self, batch: PairBatch, *, step: int, epoch: int
    ) -> tuple[torch.Tensor, Scalars]:
        logits, targets = self._forward(batch)
        loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        for m in self.metrics:
            m.update(
                outputs=logits.detach().unsqueeze(-1),
                targets=targets.detach().unsqueeze(-1),
            )
        return loss, {"loss/bce": float(loss.detach().item())}
