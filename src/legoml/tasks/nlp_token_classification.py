from __future__ import annotations

from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn

from legoml.core.contracts.task import Task, Scalars
from legoml.core.contracts.metric import Metric
from legoml.data.batches import NERBatch


@dataclass
class NERTokenClassificationTask(Task):
    model: torch.nn.Module  # module that returns logits (B, T, C)
    metrics: List[Metric]
    device: str
    ignore_index: int = -100
    _loss_fn: nn.Module | None = None

    def on_epoch_start(self, *, mode: str, epoch: int) -> None:
        for m in self.metrics:
            m.reset()

    def on_epoch_end(self, *, mode: str, epoch: int) -> Scalars:
        scalars: Scalars = {}
        for m in self.metrics:
            scalars.update(m.compute())
        return {f"{mode}/{k}": float(v) for k, v in scalars.items()}

    def _step(self, batch: NERBatch) -> tuple[torch.Tensor, Scalars, torch.Tensor]:
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        labels = batch.labels.to(self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)  # (B, T, C)
        # Cross-entropy over time steps with ignore_index
        if self._loss_fn is None:
            self._loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        # reshape: (B*T, C) vs (B*T)
        loss = self._loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        for m in self.metrics:
            m.update(outputs=logits.detach().cpu(), targets=labels.detach().cpu())
        return loss, {"loss/xent": float(loss.detach().item())}, logits

    def train_step(self, batch: NERBatch, *, step: int, epoch: int):
        return self._step(batch)[:2]

    def eval_step(self, batch: NERBatch, *, step: int, epoch: int):
        return self._step(batch)[:2]
