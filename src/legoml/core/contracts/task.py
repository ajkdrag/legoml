from typing import Protocol, runtime_checkable, Any
from legoml.core.constants import Mode
import torch


Scalars = dict[str, float]


@runtime_checkable
class Task(Protocol):
    """
    Minimal contract for a training task.
    Owns domain logic: how to turn a batch into loss + scalars + manage metrics.
    """

    model: torch.nn.Module

    def on_epoch_start(self, *, mode: Mode, epoch: int) -> None: ...
    def on_epoch_end(self, *, mode: Mode, epoch: int) -> Scalars: ...

    def train_step(
        self,
        batch: Any,
        *,
        step: int,
        epoch: int,
    ) -> tuple[torch.Tensor, Scalars]: ...

    def eval_step(
        self,
        batch: Any,
        *,
        step: int,
        epoch: int,
    ) -> tuple[torch.Tensor, Scalars]: ...
