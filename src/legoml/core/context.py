from dataclasses import dataclass, field
from typing import Callable, Any
import torch


@dataclass(kw_only=True)
class Context:
    config: Any
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer | None = None
    loss_fn: Callable
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    scaler: torch.GradScaler | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    def get_lr(self):
        if self.scheduler:
            return self.scheduler.get_last_lr()
        if self.optimizer:
            return self.optimizer.param_groups[-1]["lr"]

    def to_dict(self):
        return {
            "model": self.model.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
            "loss_fn": self.loss_fn.__class__.__name__,
            "device": self.device.type,
            "scaler": self.scaler.__class__.__name__,
            "scheduler": self.scheduler.__class__.__name__,
        }
