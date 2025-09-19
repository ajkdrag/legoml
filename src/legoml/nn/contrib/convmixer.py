from functools import partial

import torch.nn as nn

from legoml.nn.contrib.resnet import ResNetDShortcut
from legoml.nn.conv import Conv1x1ActNorm, ConvActNorm, DWConvActNorm
from legoml.nn.ops import SpaceToDepth
from legoml.nn.struct import ResidualAdd
from legoml.nn.types import ModuleCtor


class ConvMixerStem(nn.Sequential):
    def __init__(
        self,
        *,
        c_in: int,
        c_out: int | None = None,
        patch_sz: int,
        conv: ModuleCtor = partial(ConvActNorm, act=nn.GELU),
    ):
        super().__init__()
        c_out = c_out or c_in
        self.conv = conv(c_in=c_in, c_out=c_out, k=patch_sz, s=patch_sz)


class SAConvMixerStem(nn.Sequential):
    def __init__(
        self,
        *,
        c_in: int,
        c_out: int | None = None,
        patch_sz: int,
        conv: ModuleCtor = partial(Conv1x1ActNorm, act=nn.GELU),
    ):
        super().__init__()
        c_out = c_out or c_in
        self.s2d = SpaceToDepth(f_reduce=patch_sz)
        self.conv = conv(c_in=c_in * (patch_sz**2), c_out=c_out)


class ConvMixerBlock(nn.Sequential):
    def __init__(
        self,
        *,
        c_in: int,
        c_out: int | None = None,
        k: int = 5,
        s: int = 1,
        block1: ModuleCtor = partial(DWConvActNorm, act=nn.GELU),
        residual1: ModuleCtor = ResidualAdd,
        shortcut1: ModuleCtor = ResNetDShortcut,
        block2: ModuleCtor = partial(Conv1x1ActNorm, act=nn.GELU),
        residual2: ModuleCtor = partial(ResidualAdd, res_func=None),
        shortcut2: ModuleCtor = nn.Identity,
    ):
        super().__init__()
        c_out = c_out or c_in
        self.block1 = residual1(
            block=block1(c_in=c_in, k=k, s=s),
            shortcut=shortcut1(c_in=c_in, c_out=c_in, s=s),
        )
        self.block2 = residual2(
            block=block2(c_in=c_in, c_out=c_out, s=1),
            shortcut=shortcut2(c_int=c_in, c_out=c_out, s=1),
        )
