import math
from functools import partial

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from legoml.nn.primitives import (
    ModuleCtor,
    identity,
)
from legoml.nn.utils import autopad


class PoolShortcut(nn.Sequential):
    def __init__(self, k: int = 3, s: int = 1, pool: ModuleCtor = nn.AvgPool2d):
        super().__init__()

        self.shortcut = (
            identity
            if s == 1
            else pool(
                kernel_size=k,
                stride=s,
                padding=autopad(k),
            )
        )


class FCNormAct(nn.Sequential):
    """Linear layer with normalization and activation.

    Linear->BN->Act with Dropout support.

    Parameters
    ----------
    c_in : int
        Input features
    c_out : int, optional
        Output features. Defaults to c_in
    dropout : float, default=0.0
        Dropout probability after activation
    norm : Callable, optional
        Normalization. Defaults to BatchNorm2d
    act: Callable, optional
        Activation. Defaults to Relu
    """

    def __init__(
        self,
        *,
        c_in: int,
        c_out: int | None = None,
        norm: ModuleCtor = nn.BatchNorm1d,
        act: ModuleCtor = partial(nn.ReLU, inplace=True),
        dropout: float = 0.0,
    ):
        super().__init__()
        c_out = c_out or c_in

        # TODO: Should bias be set to False like ConvLayer?
        self.block = nn.Linear(c_in, c_out)
        self.norm = norm(c_out)
        self.act = act()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else identity


class ConvNormAct(nn.Sequential):
    """Convolution layer with normalization and activation (Conv->BN->Act)

    Parameters
    ----------
    c_in : int
        Input channels
    c_out : int, optional
        Output channels. Defaults to c_in
    k : int, default=3
        Kernel size
    s : int, default=1
        Stride
    p : int, optional
        Padding. Automatically computed for 'same' padding if None
    g : int, default=1
        Groups for grouped convolution
    d : int, default=1
        Dilation
    dropout : float, default=0.0
        Dropout probability after activation
    norm : Callable, optional
        Normalization. Defaults to Batchnorm2d
    act : Callable, optional
        Activation. Defaults to Relu
    **kwargs
        Additional arguments passed to Conv2d
    """

    def __init__(
        self,
        *,
        c_in: int,
        c_out: int | None = None,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        norm: ModuleCtor = nn.BatchNorm2d,
        act: ModuleCtor = partial(nn.ReLU, inplace=True),
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        c_out = c_out or c_in
        p = autopad(k, p, d)

        self.block = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            dilation=d,
            bias=False,
            **kwargs,
        )
        self.norm = norm(c_out)
        self.act = act()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else identity


class NormActConv(nn.Sequential):
    """Convolution layer with normalization and activation (BN->Act->Conv)

    Parameters
    ----------
    c_in : int
        Input channels
    c_out : int, optional
        Output channels. Defaults to c_in
    k : int, default=3
        Kernel size
    s : int, default=1
        Stride
    p : int, optional
        Padding. Automatically computed for 'same' padding if None
    g : int, default=1
        Groups for grouped convolution
    d : int, default=1
        Dilation
    dropout : float, default=0.0
        Dropout probability after activation
    norm : Callable, optional
        Normalization. Defaults to Batchnorm2d
    act : Callable, optional
        Activation. Defaults to Relu
    **kwargs
        Additional arguments passed to Conv2d
    """

    def __init__(
        self,
        *,
        c_in: int,
        c_out: int | None = None,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        norm: ModuleCtor = nn.BatchNorm2d,
        act: ModuleCtor = partial(nn.ReLU, inplace=True),
        dropout=0.0,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        c_out = c_out or c_in
        p = autopad(k, p, d)

        self.norm = norm(c_in)
        self.act = act()
        self.block = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            dilation=d,
            bias=bias,
            **kwargs,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else identity


class DWConvNormAct(ConvNormAct):
    """Depthwise convolution for efficient spatial processing.

    Applies one filter per input channel, drastically reducing parameters and
    computation compared to standard convolution. Core component of depthwise
    separable convolutions in mobile architectures.

    Parameters
    ----------
    c_in : int
        Number of input/output channels
    k : int, default=3
        Kernel size
    s : int, default=1
        Stride
    p : int, optional
        Padding. Automatically computed for 'same' padding if None
    d : int, default=1
        Dilation
    dropout : float, default=0.0
        Dropout probability after activation
    norm : Callable, optional
        Normalization. Defaults to Batchnorm2d
    act : Callable, optional
        Activation. Defaults to Relu
    **kwargs
        Additional arguments passed to Conv2d
    """

    def __init__(
        self,
        *,
        c_in: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        d: int = 1,
        norm: ModuleCtor = nn.BatchNorm2d,
        act: ModuleCtor = partial(nn.ReLU, inplace=True),
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            c_in=c_in,
            c_out=c_in,
            k=k,
            s=s,
            p=p,
            g=c_in,
            d=d,
            norm=norm,
            act=act,
            dropout=dropout,
            **kwargs,
        )


class DWSepConvNormAct(nn.Sequential):
    """Complete depthwise separable convolution.

    Combines depthwise convolution (spatial filtering) with pointwise
    convolution (channel mixing). Achieves similar representational capacity
    as standard convolution with significantly fewer parameters.

    Parameters
    ----------
    c_in : int
        Input channels
    c_out : int
        Output channels
    k : int, default=3
        Kernel size for depthwise convolution
    s : int, default=1
        Stride for depthwise convolution
    p : int, optional
        Padding for depthwise convolution
    d : int, default=1
        Dilation for depthwise convolution
    dropout : float, default=0.0
        Dropout probability
    norm : Callable, optional
        Normalization function. Defaults to batchnorm2d
    dw_act : Callable, optional
        Activation for depthwise conv. Defaults to relu
    pw_act : Callable, optional
        Activation for pointwise conv. Defaults to noop
    """

    def __init__(
        self,
        *,
        c_in: int,
        c_out: int | None = None,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        d: int = 1,
        dropout=0.0,
        norm: ModuleCtor = nn.BatchNorm2d,
        dw_act: ModuleCtor = partial(nn.ReLU, inplace=True),
        pw_act: ModuleCtor = nn.Identity,
    ):
        super().__init__()
        c_out = c_out or c_in
        self.dw_block = DWConvNormAct(
            c_in=c_in,
            s=s,
            k=k,
            p=p,
            d=d,
            norm=norm,
            act=dw_act,
        )
        self.pw_block = Conv1x1NormAct(
            c_in=c_in,
            c_out=c_out,
            dropout=dropout,
            norm=norm,
            act=pw_act,
        )


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


class GlobalAvgPool2d(nn.Sequential):
    """Global average pooling for spatial dimension reduction.

    Reduces spatial dimensions (H, W) to (1, 1) via adaptive average pooling.
    Commonly used as final pooling layer in CNNs before classification head.

    Parameters
    ----------
    keep_dim : bool, default=False
        If True, keeps spatial dimensions as (1, 1). If False, flattens to 1D
    """

    def __init__(self, keep_dim: bool = False):
        super().__init__()
        self.block = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten() if not keep_dim else identity


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


Conv3x3NormAct = partial(ConvNormAct, k=3)
Conv1x1NormAct = partial(ConvNormAct, k=1)
NormActConv3x3 = partial(NormActConv, k=3)
