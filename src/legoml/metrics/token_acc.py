from __future__ import annotations

from dataclasses import dataclass
import torch

from legoml.core.contracts.metric import Metric


@dataclass
class TokenAccuracy(Metric):
    name: str = "token_accuracy"
    ignore_index: int = -100

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, *, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        # outputs: (B, T, C) logits; targets: (B, T)
        preds = outputs.argmax(dim=-1)
        mask = targets != self.ignore_index
        if mask.sum() == 0:
            return
        self.correct += (preds[mask] == targets[mask]).sum().item()
        self.total += mask.sum().item()

    def compute(self) -> dict[str, float]:
        acc = self.correct / self.total if self.total > 0 else 0.0
        return {self.name: float(acc)}
