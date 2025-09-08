from functools import partial

import torch.nn as nn

from legoml.nn.conv import (
    Conv1x1NormAct,
    Conv3x3NormAct,
    DWSepConvNormAct,
)
from legoml.nn.struct import ResidualAdd
from legoml.nn.types import ModuleCtor
from legoml.nn.utils import identity, make_divisible

relu6 = partial(nn.ReLU6, inplace=True)


class MBConv(nn.Sequential):
    def __init__(
        self,
        *,
        c_in: int,
        c_mid: int | None = None,
        c_out: int | None = None,
        s: int = 1,
        f_expand: int = 4,
        block1: ModuleCtor = partial(Conv1x1NormAct, act=relu6),
        block2: ModuleCtor = partial(DWSepConvNormAct, dw_act=relu6),
        shortcut: ModuleCtor = nn.Identity,
        residual: ModuleCtor = ResidualAdd,
        act: ModuleCtor = relu6,
    ):
        super().__init__()
        c_out = c_out or c_in
        c_mid = c_mid or make_divisible(c_in * f_expand)

        use_expand = c_mid != c_in
        apply_residual = s == 1 and c_in == c_out

        block1 = (
            block1(
                c_in=c_in,
                c_out=c_mid,
                s=1,
            )
            if use_expand
            else identity
        )
        block2 = block2(c_in=c_mid, c_out=c_out, s=s)
        block = nn.Sequential(block1, block2)

        if apply_residual:
            block = residual(
                block=block,
                shortcut=shortcut(c_in=c_in, c_out=c_out, s=s),
            )

        self.block = block
        self.act = act()


class FusedMBConv(nn.Sequential):
    def __init__(
        self,
        *,
        c_in: int,
        c_mid: int | None = None,
        c_out: int | None = None,
        s: int = 1,
        f_expand: int = 4,
        block1: ModuleCtor = partial(Conv3x3NormAct, act=relu6),
        block2: ModuleCtor = partial(Conv1x1NormAct, act=nn.Identity),
        shortcut: ModuleCtor = nn.Identity,
        residual: ModuleCtor = ResidualAdd,
        act: ModuleCtor = relu6,
    ):
        super().__init__()
        c_out = c_out or c_in
        c_mid = c_mid or make_divisible(c_in * f_expand)

        apply_residual = s == 1 and c_in == c_out

        block1 = block1(c_in=c_in, c_out=c_mid, s=s)
        block2 = block2(c_in=c_mid, c_out=c_out, s=1)
        block = nn.Sequential(block1, block2)

        if apply_residual:
            block = residual(
                block=block,
                shortcut=shortcut(c_in=c_in, c_out=c_out, s=s),
            )

        self.block = block
        self.act = act()
