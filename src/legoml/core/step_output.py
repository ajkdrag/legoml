from dataclasses import dataclass
from typing import Any
import torch


@dataclass(kw_only=True)
class StepOutput:
    loss: torch.Tensor | None = None
    predictions: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

