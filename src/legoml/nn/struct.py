from functools import partial

import torch
import torch.nn as nn

from legoml.nn.ops import LayerScale
from legoml.nn.regularization import DropPath
from legoml.nn.types import ModuleCtor
from legoml.nn.utils import identity, make_divisible


class ApplyAfterCtor:
    """
    A factory class that creates an nn.Sequential module by instantiating
    a main block and then an 'after' block.
    """

    def __init__(self, main: ModuleCtor, after: ModuleCtor):
        self.main = main
        self.after = after

    def __call__(self, *args, **kwargs):
        main_block = self.main(*args, **kwargs)
        # Note: This assumes the 'after' block doesn't need the same args
        after_block = self.after()

        return nn.Sequential(main_block, after_block)


class Bottleneck(nn.Sequential):
    def __init__(
        self,
        *,
        c_in: int,
        c_mid: int | None = None,
        c_out: int | None = None,
        s: int = 1,
        f_reduce: int = 4,
        block1: ModuleCtor,
        block2: ModuleCtor,
        block3: ModuleCtor,
        shortcut: ModuleCtor,
        act: ModuleCtor = partial(nn.ReLU, inplace=True),
        drop_path: float = 0.0,
    ):
        super().__init__()

        c_out = c_out or c_in
        c_mid = c_mid or make_divisible(c_out / f_reduce)

        block1 = block1(c_in=c_in, c_out=c_mid, s=1)
        block2 = block2(c_in=c_mid, c_out=c_mid, s=s)  # ResNet-D style stride
        block3 = block3(c_in=c_mid, c_out=c_out, s=1)
        shortcut = shortcut(c_in=c_in, c_out=c_out, s=s)

        self.block = ScaledResidual(
            fn=nn.Sequential(block1, block2, block3),
            shortcut=shortcut,
            drop_prob=drop_path,
        )
        self.act = act()


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


class ScaledResidual(nn.Module):
    """Residual connection with stochastic depth.

    Implements residual connection with optional DropPath for
    stochastic depth regularization during training.

    Parameters
    ----------
    fn : nn.Module
        Main branch function/block
    shortcut : nn.Module, optional
        Shortcut connection. Defaults to identity
    drop_prob : float, default=0.0
        Drop path probability for stochastic depth
    """

    def __init__(
        self,
        *,
        fn: nn.Module,
        shortcut: nn.Module | None = None,
        drop_prob: float = 0.0,
        layer_scale_init: float = 0.0,
        layer_scale_dimensions: int | None = None,
    ):
        super().__init__()
        layers = [fn]
        if layer_scale_init > 0.0:
            if layer_scale_dimensions is None:
                raise ValueError(
                    "layer_scale_dimensions must be specified if layer_scale_init > 0.0"
                )
            layers.append(
                LayerScale(
                    layer_scale_init,
                    layer_scale_dimensions,
                )
            )
        if drop_prob > 0.0:
            layers.append(DropPath(drop_prob))

        self.block = nn.Sequential(*layers)
        self.shortcut = shortcut or identity

    def forward(self, x):
        return self.shortcut(x) + self.block(x)
