from __future__ import annotations

from dataclasses import dataclass
import torch

from legoml.core.contracts.metric import Metric, Scalars


@dataclass
class MultiClassAccuracy(Metric):
    name: str = "accuracy"

    def reset(self) -> None:
        self.correct = 0.0
        self.total = 0.0

    def update(self, *, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        preds = outputs.argmax(dim=1)
        self.correct += (preds == targets).sum().item()
        self.total += targets.numel()

    def compute(self) -> Scalars:
        result = self.correct / self.total if self.total else 0.0
        return {self.name: result}
