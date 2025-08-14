from dataclasses import dataclass

import torch

from legoml.core.base import OptimizerNode


@dataclass
class AdamNode(OptimizerNode):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def build(self, params) -> torch.optim.Adam:
        return torch.optim.Adam(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )


@dataclass
class SGDNode(OptimizerNode):
    """SGD optimizer configuration."""

    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    nesterov: bool = False

    def build(self, params) -> torch.optim.SGD:
        return torch.optim.SGD(
            params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )
