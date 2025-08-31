import math
from typing import Callable
from collections import OrderedDict

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from legoml.nn.activations import relu_fn
from legoml.nn.norms import bn1d_fn, bn2d_fn


def autopad(k: int, p: int | None = None, d: int = 1):
    """Utility for padding to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1
    return p or (k - 1) // 2


def make_divisible(
    v: float,
    divisor: int = 8,
    min_value: int | None = None,
    round_down_protect: bool = True,
) -> int:
    """
    Rounds a value `v` to the nearest multiple of `divisor`. This is crucial for
    optimizing performance on hardware accelerators like GPUs.
    From the original TensorFlow repository:
    https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * v:
        new_value += divisor
    return int(new_value)


identity = nn.Identity()


def noop_fn(*_args, **_kwargs):
    return identity


class NormAct(nn.Module):
    """Sequential normalization and activation layer.

    Parameters
    ----------
    norm : nn.Module
        Normalization layer (e.g., BatchNorm, LayerNorm)
    act : nn.Module
        Activation function (e.g., ReLU, GELU)
    """

    def __init__(self, norm: nn.Module, act: nn.Module):
        super().__init__()
        self.norm = norm
        self.act = act

    def forward(self, x: torch.Tensor):
        return self.act(self.norm(x))


class LinearLayer(nn.Module):
    """Linear layer with normalization and activation.

    Supports both post-normact (Linear->BN->Act) and pre-normact (BN->Act->Linear)
    patterns for improved gradient flow.

    Parameters
    ----------
    c1 : int
        Input features
    c2 : int, optional
        Output features. Defaults to c1
    dropout : float, default=0.0
        Dropout probability after activation
    norm_fn : Callable, optional
        Normalization function. Defaults to bn1d_fn
    act_fn : Callable, optional
        Activation function. Defaults to relu_fn
    pre_normact : bool, default=False
        If True, applies norm->act->linear instead of linear->norm->act
    """

    def __init__(
        self,
        *,
        c1,
        c2=None,
        dropout=0.0,
        norm_fn: Callable[..., nn.Module] | None = None,
        act_fn: Callable[..., nn.Module] | None = None,
        pre_normact=False,
    ):
        super().__init__()
        c2 = c2 or c1
        norm_fn = norm_fn or bn1d_fn
        act_fn = act_fn or relu_fn

        # TODO: Should bias be set to False like ConvLayer?
        self.linear = nn.Linear(c1, c2)
        self.norm_act = NormAct(
            norm=norm_fn(c1 if pre_normact else c2),
            act=act_fn(),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.pre_normact = pre_normact

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_normact:
            x = self.norm_act(x)
            x = self.linear(x)
        else:
            x = self.linear(x)
            x = self.norm_act(x)

        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ConvLayer(nn.Module):
    """Convolution layer with normalization and activation.

    Supports both post-normact (Conv->BN->Act) and pre-normact (BN->Act->Conv)
    patterns with automatic same padding.

    Parameters
    ----------
    c1 : int
        Input channels
    c2 : int, optional
        Output channels. Defaults to c1
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
    norm_fn : Callable, optional
        Normalization function. Defaults to bn2d_fn
    act_fn : Callable, optional
        Activation function. Defaults to relu_fn
    pre_normact : bool, default=False
        If True, applies norm->act->conv instead of conv->norm->act
    **kwargs
        Additional arguments passed to Conv2d
    """

    def __init__(
        self,
        *,
        c1: int,
        c2: int | None = None,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        dropout=0.0,
        norm_fn: Callable[..., nn.Module] | None = None,
        act_fn: Callable[..., nn.Module] | None = None,
        pre_normact=False,
        **kwargs,
    ):
        super().__init__()
        c2 = c2 or c1
        p = autopad(k, p, d)
        norm_fn = norm_fn or bn2d_fn
        act_fn = act_fn or relu_fn

        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            dilation=d,
            bias=True if pre_normact else False,
            **kwargs,
        )
        self.norm_act = NormAct(
            norm=norm_fn(c1 if pre_normact else c2),
            act=act_fn(),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.pre_normact = pre_normact

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_normact:
            x = self.norm_act(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.norm_act(x)

        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Conv1x1(ConvLayer):
    """1x1 convolution layer for channel-wise transformations.

    Pointwise convolution used for channel mixing, dimension reduction/expansion,
    and feature transformations without spatial interactions.

    Parameters
    ----------
    c1 : int
        Input channels
    c2 : int, optional
        Output channels. Defaults to c1
    s : int, default=1
        Stride
    p : int, optional
        Padding
    g : int, default=1
        Groups for grouped convolution
    d : int, default=1
        Dilation
    dropout : float, default=0.0
        Dropout probability after activation
    norm_fn : Callable, optional
        Normalization function. Defaults to bn2d_fn
    act_fn : Callable, optional
        Activation function. Defaults to relu_fn
    pre_normact : bool, default=False
        If True, applies norm->act->conv instead of conv->norm->act
    **kwargs
        Additional arguments passed to Conv2d
    """

    def __init__(
        self,
        *,
        c1: int,
        c2: int | None = None,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        dropout=0.0,
        norm_fn: Callable[..., nn.Module] | None = None,
        act_fn: Callable[..., nn.Module] | None = None,
        pre_normact=False,
        **kwargs,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            k=1,
            s=s,
            p=p,
            g=g,
            d=d,
            dropout=dropout,
            norm_fn=norm_fn,
            act_fn=act_fn,
            pre_normact=pre_normact,
            **kwargs,
        )


class Conv3x3(ConvLayer):
    """3x3 convolution layer for spatial feature extraction.

    Standard spatial convolution with 3x3 kernel for local feature extraction
    and pattern recognition in computer vision tasks.

    Parameters
    ----------
    c1 : int
        Input channels
    c2 : int, optional
        Output channels. Defaults to c1
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
    norm_fn : Callable, optional
        Normalization function. Defaults to bn2d_fn
    act_fn : Callable, optional
        Activation function. Defaults to relu_fn
    pre_normact : bool, default=False
        If True, applies norm->act->conv instead of conv->norm->act
    **kwargs
        Additional arguments passed to Conv2d
    """

    def __init__(
        self,
        *,
        c1: int,
        c2: int | None = None,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        dropout=0.0,
        norm_fn: Callable[..., nn.Module] | None = None,
        act_fn: Callable[..., nn.Module] | None = None,
        pre_normact=False,
        **kwargs,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            k=3,
            s=s,
            p=p,
            g=g,
            d=d,
            dropout=dropout,
            norm_fn=norm_fn,
            act_fn=act_fn,
            pre_normact=pre_normact,
            **kwargs,
        )


class Shortcut(nn.Module):
    """Identity or projection shortcut for residual connections.

    Automatically handles dimension matching for residual connections by using
    identity when input/output dimensions match, otherwise applies 1x1 conv
    projection to match channels and/or spatial dimensions with stride.

    Parameters
    ----------
    c1 : int
        Input channels
    c2 : int
        Output channels
    s : int, default=1
        Stride for projection when needed
    p : int, optional
        Padding for projection convolution
    g : int, default=1
        Groups for projection convolution
    d : int, default=1
        Dilation for projection convolution
    dropout : float, default=0.0
        Dropout probability
    norm_fn : Callable, optional
        Normalization function for projection
    act_fn : Callable, optional
        Activation function for projection. Defaults to noop_fn
    pre_normact : bool, default=False
        If True, uses pre-activation pattern for projection
    """

    def __init__(
        self,
        *,
        c1: int,
        c2: int,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        dropout=0.0,
        norm_fn: Callable[..., nn.Module] | None = None,
        act_fn: Callable[..., nn.Module] | None = None,
        pre_normact: bool = False,
    ):
        super().__init__()
        act_fn = act_fn or noop_fn
        self.shortcut = (
            identity
            if c1 == c2 and s == 1
            else Conv1x1(
                c1=c1,
                c2=c2,
                s=s,
                p=p,
                g=g,
                d=d,
                dropout=dropout,
                norm_fn=norm_fn,
                act_fn=act_fn,
                pre_normact=pre_normact,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x)


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


class GlobalAvgPool2d(nn.Module):
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
        self.keep_dim = keep_dim
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten() if not keep_dim else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        if self.flatten is not None:
            x = self.flatten(x)
        return x


class ChannelShuffle(nn.Module):
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
        self.rearrange = Rearrange("b (g c_per_g) h w -> b (c_per_g g) h w", g=g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rearrange(x)


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


class DWConv(ConvLayer):
    """Depthwise convolution for efficient spatial processing.

    Applies one filter per input channel, drastically reducing parameters and
    computation compared to standard convolution. Core component of depthwise
    separable convolutions in mobile architectures.

    Parameters
    ----------
    c : int
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
    norm_fn : Callable, optional
        Normalization function. Defaults to bn2d_fn
    act_fn : Callable, optional
        Activation function. Defaults to relu_fn
    pre_normact : bool, default=False
        If True, applies norm->act->conv instead of conv->norm->act
    **kwargs
        Additional arguments passed to Conv2d
    """

    def __init__(
        self,
        *,
        c: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        d: int = 1,
        dropout=0.0,
        norm_fn: Callable[..., nn.Module] | None = None,
        act_fn: Callable[..., nn.Module] | None = None,
        pre_normact=False,
        **kwargs,
    ):
        super().__init__(
            c1=c,
            c2=c,
            k=k,
            s=s,
            p=p,
            g=c,
            d=d,
            dropout=dropout,
            norm_fn=norm_fn,
            act_fn=act_fn,
            pre_normact=pre_normact,
            **kwargs,
        )


class DWSepConv(nn.Module):
    """Complete depthwise separable convolution.

    Combines depthwise convolution (spatial filtering) with pointwise
    convolution (channel mixing). Achieves similar representational capacity
    as standard convolution with significantly fewer parameters.

    Parameters
    ----------
    c1 : int
        Input channels
    c2 : int
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
    norm_fn : Callable, optional
        Normalization function. Defaults to bn2d_fn
    dw_act_fn : Callable, optional
        Activation for depthwise conv. Defaults to relu_fn
    pw_act_fn : Callable, optional
        Activation for pointwise conv. Defaults to noop_fn
    pre_normact : bool, default=False
        If True, applies norm->act->conv pattern
    """

    def __init__(
        self,
        *,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        d: int = 1,
        dropout=0.0,
        norm_fn: Callable[..., nn.Module] | None = None,
        dw_act_fn: Callable[..., nn.Module] | None = None,
        pw_act_fn: Callable[..., nn.Module] | None = None,
        pre_normact=False,
    ):
        super().__init__()
        pw_act_fn = pw_act_fn or noop_fn
        self.dw_conv = DWConv(
            c=c1,
            s=s,
            k=k,
            p=p,
            d=d,
            dropout=dropout,
            norm_fn=norm_fn,
            act_fn=dw_act_fn,
            pre_normact=pre_normact,
        )
        self.pw_conv = Conv1x1(
            c1=c1,
            c2=c2,
            dropout=dropout,
            norm_fn=norm_fn,
            act_fn=pw_act_fn,
            pre_normact=pre_normact,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x
