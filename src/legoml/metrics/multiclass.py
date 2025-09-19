from dataclasses import dataclass

import torch

from legoml.core.metric import Metric


@dataclass
class MultiClassAccuracy(Metric):
    name: str = "accuracy"

    def reset(self) -> None:
        self.correct = 0.0
        self.total = 0.0

    def update(self, output) -> None:
        assert isinstance(output.predictions, torch.Tensor)
        assert isinstance(output.targets, torch.Tensor)

        preds = output.predictions.argmax(dim=1)
        self.correct += (preds == output.targets).sum().item()
        self.total += output.targets.numel()

    def compute(self):
        result = self.correct / self.total if self.total else 0.0
        return {self.name: result}
