from dataclasses import dataclass
import torch

from legoml.core.contracts.metric import Metric, Scalars


@dataclass
class BinaryF1(Metric):
    name: str = "f1"
    threshold: float = 0.5
    apply_sigmoid: bool = True

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, *, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        probs = outputs
        if self.apply_sigmoid:
            probs = torch.sigmoid(outputs)
        preds = (probs >= self.threshold).long().view_as(targets)
        t = targets.long()
        self.tp += int(((preds == 1) & (t == 1)).sum().item())
        self.fp += int(((preds == 1) & (t == 0)).sum().item())
        self.fn += int(((preds == 0) & (t == 1)).sum().item())

    def compute(self) -> Scalars:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        return {
            self.name: float(f1),
            f"{self.name}/precision": float(precision),
            f"{self.name}/recall": float(recall),
        }


@dataclass
class MacroF1(Metric):
    name: str = "f1_macro"
    num_classes: int = 2

    def reset(self) -> None:
        self.preds = []
        self.targets = []

    def update(self, *, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        preds = outputs.argmax(dim=1)
        self.preds.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> Scalars:
        if not self.preds:
            return {self.name: 0.0}
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)

        f1_sum = 0.0
        for _cls in range(self.num_classes):
            tp = ((preds == _cls) & (targets == _cls)).sum().item()
            fp = ((preds == _cls) & (targets != _cls)).sum().item()
            fn = ((preds != _cls) & (targets == _cls)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                (2 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else 0.0
            )
            f1_sum += f1
        return {self.name: float(f1_sum / max(1, self.num_classes))}
