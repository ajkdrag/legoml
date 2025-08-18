from typing import Protocol, Dict, Any
from legoml.core.constants import Mode
import torch

Scalars = Dict[str, float]


class Objective(Protocol):
    """Compute a loss and optional scalars given model + batch + mode."""

    def __call__(
        self,
        model: torch.nn.Module,
        batch: Any,
        *,
        mode: Mode,
    ) -> tuple[torch.Tensor, Scalars]: ...
