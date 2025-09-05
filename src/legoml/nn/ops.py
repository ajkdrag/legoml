import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from legoml.nn.regularization import DropPath
from legoml.nn.utils import identity


class BranchAndConcat(nn.Module):
    """Parallel branches with channel-wise concatenation.

    Applies multiple branches in parallel and concatenates their outputs
    along the channel dimension. Used in Inception-style architectures.

    Parameters
    ----------
    *branches : nn.Module
        Variable number of branch modules to apply in parallel
    """

    def __init__(self, *branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class ChannelShuffle(nn.Sequential):
    """Channel shuffling for grouped convolution information exchange.

    Permutes channels to enable information flow between groups in grouped
    convolutions. Essential for maintaining representational capacity in
    efficient architectures like ShuffleNet.

    Parameters
    ----------
    g : int
        Number of groups for channel shuffling
    """

    def __init__(self, g):
        super().__init__()
        self.block = Rearrange("b (g c_per_g) h w -> b (c_per_g g) h w", g=g)


class ScaledResidual(nn.Module):
    """Residual connection with learnable scaling and stochastic depth.

    Implements residual connection with learnable alpha scaling parameter and
    optional DropPath for stochastic depth regularization during training.

    Parameters
    ----------
    fn : nn.Module
        Main branch function/block
    shortcut : nn.Module, optional
        Shortcut connection. Defaults to identity
    drop_prob : float, default=0.0
        Drop path probability for stochastic depth
    alpha_init : float, default=1.0
        Initial value for learnable scaling parameter
    """

    def __init__(
        self,
        *,
        fn: nn.Module,
        shortcut: nn.Module | None = None,
        drop_prob: float = 0.0,
        alpha_init: float = 1.0,
    ):
        super().__init__()
        self.fn = fn
        self.shortcut = shortcut or identity
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else identity
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x):
        out = self.fn(x)
        out = self.drop_path(out)
        return self.shortcut(x) + self.alpha * out
