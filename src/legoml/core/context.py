from dataclasses import dataclass, field
from typing import Callable, Any
import torch


@dataclass(kw_only=True)
class Context:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer | None = None
    loss_fn: Callable
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    config: Any | None = None
    scaler: torch.GradScaler | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
