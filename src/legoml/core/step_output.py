from dataclasses import dataclass
from typing import Any
import torch


@dataclass(kw_only=True)
class StepOutput:
    loss: torch.Tensor | None = None
    predictions: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    @property
    def loss_scalar(self) -> float | None:
        if self.loss is None:
            return None
        return self.loss.item()
