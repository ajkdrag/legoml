from functools import partial

import torch.nn as nn

from legoml.nn.activation import LayerNorm2d
from legoml.nn.blocks.resnet import ResNetShortcut
from legoml.nn.conv import Conv1x1, Conv1x1NormAct, DWConv, NormActConv
from legoml.nn.struct import ScaledResidual
from legoml.nn.types import ModuleCtor
from legoml.nn.utils import make_divisible


class ConvNeXtDownsample(nn.Sequential):
    def __init__(
        self,
        *,
        c_in: int,
        c_out: int | None = None,
        s: int = 2,
        block: ModuleCtor = partial(NormActConv, k=2, norm=LayerNorm2d, act=None),
    ):
        super().__init__()
        c_out = c_out or c_in
        self.block = block(c_in=c_in, c_out=c_out, s=s)


class ConvNeXtBlock(nn.Sequential):
    """
    Ideally to be used with c_in == c_out and s = 1. For increasing channels
    and/or downsampling, use ConvNeXtDownsample block.
    """

    def __init__(
        self,
        *,
        c_in: int,
        c_mid: int | None = None,
        c_out: int | None = None,
        s: int = 1,
        f_expand: int = 4,
        block1: ModuleCtor = lambda *, c_in, c_out, s: DWConv(
            c_in=c_in,
            k=3,
            norm=LayerNorm2d,
            act=None,
            s=s,
        ),
        block2: ModuleCtor = partial(Conv1x1NormAct, norm=None, act=nn.GELU),
        block3: ModuleCtor = Conv1x1,
        shortcut: ModuleCtor = ResNetShortcut,
        act: ModuleCtor = nn.Identity,
        drop_path: float = 0.0,
    ):
        super().__init__()
        c_out = c_out or c_in
        c_mid = c_mid or make_divisible(c_in * f_expand)

        block1 = block1(c_in=c_in, c_out=c_in, s=s)
        block2 = block2(c_in=c_in, c_out=c_mid, s=1)
        block3 = block3(c_in=c_mid, c_out=c_out, s=1)
        shortcut = shortcut(c_in=c_in, c_out=c_out, s=s)

        self.block = ScaledResidual(
            fn=nn.Sequential(block1, block2, block3),
            shortcut=shortcut,
            drop_prob=drop_path,
        )
        self.act = act()
