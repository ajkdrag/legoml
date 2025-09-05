import math

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization.

    Randomly drops entire residual branches during training while scaling
    remaining samples to maintain expected output magnitude. Improves
    regularization and training stability in deep networks.

    Parameters
    ----------
    p : float, default=0.0
        Drop probability. 0.0 means no dropping, 1.0 means always drop
    inplace: bool, default=False
        Whether to perform the operation inplace or not
    """

    def __init__(self, p: float = 0.0, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        if not self.training or math.isclose(self.p, 0):
            return x

        keep_prob = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1,...)
        mask = torch.full(shape, keep_prob, device=x.device).bernoulli()
        mask.div_(keep_prob)
        if self.inplace:
            x.mul_(mask)
        else:
            x = x * mask
        return x
