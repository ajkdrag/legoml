from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from legoml.nn.types import ModuleCtor
from legoml.nn.utils import identity


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


class InputForward(nn.Module):
    """
    Passes the input to multiple modules and applies an aggregation on the result.
    """

    def __init__(self, blocks: nn.Sequential, agg: ModuleCtor):
        super().__init__()
        self.layers = blocks
        self.agg = agg

    def forward(self, x):
        out = None
        for block in self.layers:
            block_out = block(x)
            out = block_out if out is None else self.agg([block_out, out])
        return out


class Residual(nn.Module):
    def __init__(
        self,
        *,
        block: nn.Module,
        res_func: Callable[[Tensor, Tensor], Tensor] | None = None,
        shortcut: nn.Module = identity,
    ):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        self.res_func = res_func

    def forward(self, x):
        res = self.shortcut(x)
        x = self.block(x)
        if self.res_func is not None:
            x = self.res_func(x, res)
        return x


ResidualAdd = partial(Residual, res_func=lambda x, res: x.add(res))
