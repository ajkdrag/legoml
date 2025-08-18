from __future__ import annotations

from dataclasses import dataclass, field
import torch

from legoml.core.contracts.task import Task, Scalars
from legoml.core.constants import Device


@dataclass
class FitState:
    epoch: int = 0
    step: int = 0
    global_step: int = 0
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    task: Task | None = None
    device: Device | None = None

    # Last step scalars
    last_train: Scalars = field(default_factory=dict)
    last_eval: Scalars = field(default_factory=dict)
    last_train_loss: float | None = None
    last_eval_loss: float | None = None

    # Epoch aggregated scalars (from task metrics)
    train_epoch: Scalars = field(default_factory=dict)
    eval_epoch: Scalars = field(default_factory=dict)

    # For progress bars
    train_total: int | None = None
    eval_total: int | None = None
