from dataclasses import dataclass

import torch

from legoml.core.base import SchedulerNode


@dataclass
class StepLRNode(SchedulerNode):
    """Step learning rate scheduler configuration."""

    step_size: int = 10
    gamma: float = 0.1

    def build(self, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )


@dataclass
class CosineAnnealingNode(SchedulerNode):
    """Cosine annealing scheduler configuration."""

    T_max: int = 50
    eta_min: float = 0.0

    def build(self, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.T_max, eta_min=self.eta_min
        )
