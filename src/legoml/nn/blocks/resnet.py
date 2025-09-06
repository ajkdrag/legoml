from functools import partial

import torch
import torch.nn as nn

from legoml.nn.conv import (
    Conv1x1NormAct,
    Conv3x3NormAct,
    DWConvNormAct,
    NormActConv3x3,
)
from legoml.nn.shortcut import PoolShortcut
from legoml.nn.struct import Bottleneck, ScaledResidual
from legoml.nn.types import ModuleCtor
from legoml.nn.utils import autopad, identity


class ResNetShortcut(nn.Sequential):
    def __init__(
        self,
        *,
        c_in: int,
        c_out: int | None = None,
        s: int = 1,
        block: ModuleCtor = partial(Conv1x1NormAct, act=nn.Identity),
    ):
        super().__init__()
        c_out = c_out or c_in
        self.block = (
            identity
            if c_in == c_out and s == 1
            else block(
                c_in=c_in,
                c_out=c_out,
                s=s,
            )
        )


class ResNetDShortcut(nn.Sequential):
    def __init__(
        self,
        *,
        c_in: int,
        c_out: int | None = None,
        k: int = 3,
        s: int = 1,
        pool: ModuleCtor = nn.AvgPool2d,
        block: ModuleCtor = partial(Conv1x1NormAct, act=nn.Identity),
    ):
        super().__init__()
        c_out = c_out or c_in
        if c_in == c_out and s == 1:
            self.block = identity
        else:
            self.pool = pool(kernel_size=k, stride=s, padding=autopad(k))
            self.block = block(c_in=c_in, c_out=c_out, s=1)


class ResNetBasic(nn.Sequential):
    """Basic ResNet block with dual 3×3 convolution path.

    Architecture
    ------------
        3×3 Conv (BN, ReLU) -> 3×3 Conv (BN) -> + Shortcut -> ReLU

    Notes
    -----
    Enhanced with learnable residual scaling, and stochastic depth.

    Parameters
    ----------
    c_in : int
        Input channels
    c_out : int, optional
        Output channels. Defaults to c_in
    s : int, default=1
        Stride
    block1: Callable, optional
        First block. Defaults to Conv3x3NormAct
    block2: Callable, optional
        Second block. Defaults to Conv3x3NormAct
    shortcut: Callable, optional
        Shortcut block. Defaults to ResNetShortcut
    act : Callable, optional
        Post residual activation function. Defaults to relu
    drop_path : float, default=0.0
        Drop path probability for stochastic depth
    """

    def __init__(
        self,
        *,
        c_in: int,
        c_out: int | None = None,
        s: int = 1,
        block1: ModuleCtor = Conv3x3NormAct,
        block2: ModuleCtor = partial(
            Conv3x3NormAct,
            act=nn.Identity,
        ),
        shortcut: ModuleCtor = partial(ResNetShortcut, block=Conv1x1NormAct),
        act: ModuleCtor = partial(nn.ReLU, inplace=True),
        drop_path: float = 0.0,
    ):
        super().__init__()
        c_out = c_out or c_in

        block1 = block1(c_in=c_in, c_out=c_out, s=s)
        block2 = block2(c_in=c_out, c_out=c_out, s=1)
        shortcut = shortcut(c_in=c_in, c_out=c_out, s=s)

        self.block = ScaledResidual(
            fn=nn.Sequential(block1, block2),
            shortcut=shortcut,
            drop_prob=drop_path,
        )
        self.act = act()


class ResNetPreAct(nn.Sequential):
    """PreAct ResNet block with dual 3×3 convolution path.

    Architecture
    ------------
        BN, ReLU, 3×3 Conv -> BN, ReLU, 3×3 Conv -> + Shortcut (no act, no norm)

    Notes
    -----
    Enhanced with learnable residual scaling, and stochastic depth.

    Parameters
    ----------
    c_in : int
        Input channels
    c_out : int, optional
        Output channels. Defaults to c_in
    s : int, default=1
        Stride
    block1: Callable, optional
        First block. Defaults to NormActConv3x3
    block2: Callable, optional
        Second block. Defaults to NormActConv3x3
    shortcut: Callable, optional
        Shortcut block. Defaults to ResNetShortcut
    act : Callable, optional
        Post residual activation function. Defaults to relu
    drop_path : float, default=0.0
        Drop path probability for stochastic depth
    """

    def __init__(
        self,
        *,
        c_in: int,
        c_out: int | None = None,
        s: int = 1,
        block1: ModuleCtor = partial(NormActConv3x3, bias=False),
        block2: ModuleCtor = NormActConv3x3,
        shortcut: ModuleCtor = partial(
            ResNetShortcut,
            block=partial(Conv1x1NormAct, norm=nn.Identity, act=nn.Identity),
        ),
        act: ModuleCtor = nn.Identity,
        drop_path: float = 0.0,
    ):
        super().__init__()
        c_out = c_out or c_in

        block1 = block1(c_in=c_in, c_out=c_out, s=s)
        block2 = block2(c_in=c_out, c_out=c_out, s=1)
        shortcut = shortcut(c_in=c_in, c_out=c_out, s=s)

        self.block = ScaledResidual(
            fn=nn.Sequential(block1, block2),
            shortcut=shortcut,
            drop_prob=drop_path,
        )
        self.act = act()


class Res2NetBlock(nn.Module):
    def __init__(
        self,
        *,
        c_in: int,
        c_mid: int | None = None,
        c_out: int | None = None,
        f_split: int = 2,
        s: int = 1,
        block: ModuleCtor = Conv3x3NormAct,
        shortcut: ModuleCtor = lambda *, c_in, c_out, s: PoolShortcut(s=s),
        projection: ModuleCtor = Conv1x1NormAct,
        alpha_init: float = 1.0,
    ):
        super().__init__()
        c_out = c_out or c_in
        c_mid = c_mid or c_in // f_split
        f_split = c_in // c_mid
        assert f_split > 1, "Atleast 2 splits necessary, for 1 use ResNetBottleneck"
        assert f_split * c_mid == c_in, "c_in should be divisible by c_mid"

        self.s = s
        self.f_split = f_split
        self.shortcut = shortcut(c_in=c_mid, c_out=c_mid, s=s)
        self.blocks = nn.ModuleList(
            block(
                c_in=c_mid,
                c_out=c_mid,
                s=s,
            )
            for _ in range(f_split - 1)
        )
        self.alpha = nn.Parameter(
            torch.tensor([alpha_init] * (f_split - 1)),
        )
        self.projection = (
            identity
            if c_in == c_out
            else projection(
                c_in=c_in,
                c_out=c_out,
            )
        )

    def forward(self, x: torch.Tensor):
        splits = x.chunk(self.f_split, dim=1)
        shortcut_split, rest = splits[-1], splits[:-1]
        y = self.blocks[0](rest[0])
        outputs = [y]

        for i in range(1, len(self.blocks)):
            ip = (rest[i] + self.alpha[i] * y) if self.s == 1 else rest[i]
            y = self.blocks[i](ip)
            outputs.append(y)

        outputs.append(self.shortcut(shortcut_split))
        return self.projection(torch.cat(outputs, dim=1))


ResNetBottleneck = partial(
    Bottleneck,
    block1=Conv1x1NormAct,
    block2=Conv3x3NormAct,
    block3=partial(Conv1x1NormAct, act=nn.Identity),
    shortcut=ResNetShortcut,
)

Res2NetBottleneck = partial(
    Bottleneck,
    block1=Conv1x1NormAct,
    block2=partial(Res2NetBlock, f_split=2),
    block3=Conv1x1NormAct,
    shortcut=ResNetShortcut,
)

ResNeXtBottleneck = partial(
    Bottleneck,
    block1=Conv1x1NormAct,
    block2=DWConvNormAct,
    block3=partial(Conv1x1NormAct, act=nn.Identity),
    shortcut=ResNetShortcut,
)
