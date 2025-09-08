import torch.nn as nn

from legoml.nn.types import ModuleCtor
from legoml.nn.utils import identity


class ResidualAdd(nn.Module):
    """Residual connection with stochastic depth.

    Implements residual connection with optional DropPath for
    stochastic depth regularization during training.

    Parameters
    ----------
    fn : nn.Module
        Main branch function/block
    shortcut : nn.Module, optional
        Shortcut connection. Defaults to identity
    """

    def __init__(
        self,
        *,
        fn: nn.Module,
        shortcut: nn.Module | None = identity,
        scaler: ModuleCtor | None = None,
        regularizer: ModuleCtor | None = None,
    ):
        super().__init__()
        layers = [fn]
        self.apply_shortcut = shortcut is not None

        if scaler:
            layers.append(scaler())
        if regularizer:
            layers.append(regularizer())

        self.block = nn.Sequential(*layers)

        if shortcut:
            self.shortcut = shortcut

    def forward(self, x):
        out = self.block(x)
        if self.apply_shortcut:
            return self.shortcut(x) + out
        return out
