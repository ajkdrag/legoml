from functools import partial

import torch.nn as nn

from legoml.nn.types import ModuleCtor
from legoml.nn.utils import autopad


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
        norm: ModuleCtor | None = nn.BatchNorm2d,
        act: ModuleCtor | None = partial(nn.ReLU, inplace=True),
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
        if norm:
            self.norm = norm(c_out)
        if act:
            self.act = act()
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)


class ConvActNorm(nn.Sequential):
    """Convolution layer with normalization and activation (Conv->Act->Norm)"""

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
        norm: ModuleCtor | None = nn.BatchNorm2d,
        act: ModuleCtor | None = partial(nn.ReLU, inplace=True),
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
            bias=True,
            **kwargs,
        )
        if act:
            self.act = act()
        if norm:
            self.norm = norm(c_out)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)


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
        norm: ModuleCtor | None = nn.BatchNorm2d,
        act: ModuleCtor | None = partial(nn.ReLU, inplace=True),
        dropout=0.0,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        c_out = c_out or c_in
        p = autopad(k, p, d)

        if norm:
            self.norm = norm(c_in)
        if act:
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

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)


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
        norm: ModuleCtor | None = nn.BatchNorm2d,
        act: ModuleCtor | None = partial(nn.ReLU, inplace=True),
        dropout=0.0,
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
        )


class DWConvActNorm(ConvActNorm):
    def __init__(
        self,
        *,
        c_in: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        d: int = 1,
        norm: ModuleCtor | None = nn.BatchNorm2d,
        act: ModuleCtor | None = partial(nn.ReLU, inplace=True),
        dropout=0.0,
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
        norm: ModuleCtor | None = nn.BatchNorm2d,
        dw_act: ModuleCtor | None = partial(nn.ReLU, inplace=True),
        pw_act: ModuleCtor | None = None,
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


Conv1x1 = partial(ConvNormAct, k=1, norm=None, act=None)
Conv3x3 = partial(ConvNormAct, k=3, norm=None, act=None)
DWConv = partial(DWConvNormAct, norm=None, act=None)
Conv3x3NormAct = partial(ConvNormAct, k=3)
Conv1x1NormAct = partial(ConvNormAct, k=1)
NormActConv3x3 = partial(NormActConv, k=3)
Conv1x1ActNorm = partial(ConvActNorm, k=1)
