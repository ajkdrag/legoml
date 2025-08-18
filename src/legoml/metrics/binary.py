from dataclasses import dataclass
import torch

from legoml.core.contracts.metric import Metric, Scalars


@dataclass
class BinaryAccuracy(Metric):
    name: str = "accuracy"
    threshold: float = 0.5
    apply_sigmoid: bool = True

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, *, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        probs = outputs
        if self.apply_sigmoid:
            probs = torch.sigmoid(outputs)
        preds = (probs >= self.threshold).long().view_as(targets)
        self.correct += (preds == targets).sum().item()
        self.total += targets.numel()

    def compute(self) -> Scalars:
        acc = self.correct / self.total if self.total > 0 else 0.0
        return {self.name: float(acc)}
