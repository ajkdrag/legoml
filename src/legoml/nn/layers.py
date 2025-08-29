import math
from typing import Callable
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn


def autopad(k: int, p: int | None = None, d: int = 1):
    """Utility for padding to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1
    return p or (k - 1) // 2


identity = nn.Identity()


class NormAct(nn.Sequential):
    def __init__(
        self,
        *,
        c,
        norm_fn: Callable[..., nn.Module] = lambda c: nn.BatchNorm2d(c),
        act: nn.Module,
    ):
        super().__init__(
            norm_fn(c),
            act,
        )


class LinearLayer(nn.Sequential):
    default_act = nn.ReLU(inplace=True)

    def __init__(
        self,
        *,
        c1,
        c2=None,
        dropout=0.0,
        norm_fn: Callable[..., nn.Module] = lambda c: nn.BatchNorm1d(c),
        act: nn.Module | None = None,
        pre_normact=False,
    ):
        super().__init__()
        c2 = c2 or c1
        act = act or self.default_act

        layers = [
            # TODO: Should bias be set to False like ConvLayer?
            nn.Linear(c1, c2),
            NormAct(c=c1 if pre_normact else c2, norm_fn=norm_fn, act=act),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        super().__init__(*layers)


class ConvLayer(nn.Sequential):
    default_act = nn.ReLU(inplace=True)

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
        norm_fn: Callable[..., nn.Module] = lambda c: nn.BatchNorm2d(c),
        act: nn.Module | None = None,
        pre_normact=False,
        **kwargs,
    ):
        c2 = c2 or c1
        p = autopad(k, p, d)
        act = act or self.default_act
        layers = [
            nn.Conv2d(
                c1,
                c2,
                kernel_size=k,
                stride=s,
                padding=p,
                groups=g,
                dilation=d,
                bias=True if pre_normact else False,
                **kwargs,
            ),
            NormAct(c=c1 if pre_normact else c2, norm_fn=norm_fn, act=act),
        ]
        if pre_normact:
            layers.reverse()
        super().__init__(*layers)


class Conv1x1(ConvLayer):
    def __init__(
        self,
        *,
        c1: int,
        c2: int | None = None,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        norm_fn: Callable[..., nn.Module] = lambda c: nn.BatchNorm2d(c),
        act: nn.Module | None = None,
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
            norm_fn=norm_fn,
            act=act,
            pre_normact=pre_normact,
            **kwargs,
        )


class Conv3x3(ConvLayer):
    def __init__(
        self,
        *,
        c1: int,
        c2: int | None = None,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        norm_fn: Callable[..., nn.Module] = lambda c: nn.BatchNorm2d(c),
        act: nn.Module | None = None,
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
            norm_fn=norm_fn,
            act=act,
            pre_normact=pre_normact,
            **kwargs,
        )


class Shortcut(nn.Sequential):
    """
    Identity or Projection Shortcut for Residual Connections.
    Used in: ResNet, most residual architectures.

    Notes
    -----
    Handles dimension matching for residual connections:
    - Identity if input/output shapes match
    - 1x1 conv projection if channels differ
    - Stride if spatial dimensions differ
    """

    default_act = identity

    def __init__(
        self,
        *,
        c1: int,
        c2: int,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        norm_fn: Callable[..., nn.Module] = lambda c: nn.BatchNorm2d(c),
        act: nn.Module | None = None,
        pre_normact: bool = False,
    ):
        act = act or self.default_act
        shortcut = (
            identity
            if c1 == c2 and s == 1
            else Conv1x1(
                c1=c1,
                c2=c2,
                s=s,
                p=p,
                g=g,
                d=d,
                norm_fn=norm_fn,
                act=act,
                pre_normact=pre_normact,
            )
        )
        super().__init__(shortcut)


class ScaledResidual(nn.Module):
    """Residual Connection with DropPath and optional scaling."""

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
    def __init__(self, *branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class GlobalAvgPool2d(nn.Sequential):
    """
    Global Average Pooling with option to keep_dim or flatten
    Used in: Most modern CNNs for final feature extraction.

    Notes
    -----
    Reduces spatial dimensions to 1x1 via average pooling.
    """

    def __init__(self, keep_dim: bool = False):
        self.keep_dim = keep_dim
        layers: list[nn.Module] = [
            nn.AdaptiveAvgPool2d((1, 1)),
        ]
        if not keep_dim:
            layers.append(nn.Flatten())
        super().__init__(*layers)


class ChannelShuffle(nn.Sequential):
    """
    Permutes channels to enable information exchange between groups.
    Used in: ShuffleNet family.

    Notes
    -----
    The Einops implementation makes it easy to understand
    """

    def __init__(self, g):
        super().__init__(Rearrange("b (g c_per_g) h w -> b (c_per_g g) h w", g=g))


class DropPath(nn.Module):
    """
    Applies Stochastic Depth, or "Drop Path", to a residual branch.
    Used in: EfficientNet, RegNet, and Vision Transformers.

    During training, this module randomly sets the entire residual branch to zero
    for some samples in the batch. The remaining active samples are scaled up to
    compensate for the dropped values, which maintains the expected magnitude of
    the output and ensures stable training.

    Notes
    -----
    The scaling of the remaining values is performed by dividing by `(1 - p)`.
    This ensures that the expected value of the output remains the same, a crucial
    property for stable training.

    References
    ----------
    "Deep Networks with Stochastic Depth" (Huang et al., 2016)
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or math.isclose(self.p, 0):
            return x

        keep_prob = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1,...)
        mask = torch.full(shape, keep_prob, device=x.device).bernoulli()

        return (x * mask) / keep_prob


class DWConv3x3(ConvLayer):
    """
    Depthwise 3x3 Convolution with BatchNorm and Activation.
    Used in: MobileNet family, EfficientNet, many mobile architectures.

    Notes
    -----
    Depthwise separable convolution - applies one filter per input channel.
    Drastically reduces parameters and computation vs standard conv.

    References
    ----------
    MobileNets: Efficient Convolutional Neural Networks (2017)
    """

    def __init__(
        self,
        *,
        c: int,
        s: int = 1,
        p: int | None = None,
        d: int = 1,
        norm_fn: Callable[..., nn.Module] = lambda c: nn.BatchNorm2d(c),
        act: nn.Module | None = None,
        pre_normact=False,
        **kwargs,
    ):
        super().__init__(
            c1=c,
            c2=c,
            k=3,
            s=s,
            p=p,
            g=c,
            d=d,
            norm_fn=norm_fn,
            act=act,
            pre_normact=pre_normact,
            **kwargs,
        )


class DWSepConv3x3(nn.Sequential):
    """
    Full Depthwise Separable Convolution (DW + PW).
    Used in: MobileNet, Xception.

    Notes
    -----
    Complete depthwise separable conv: depthwise 3x3 followed by pointwise 1x1.
    """

    default_pw_act = identity

    def __init__(
        self,
        *,
        c1: int,
        c2: int,
        s: int = 1,
        p: int | None = None,
        d: int = 1,
        norm_fn: Callable[..., nn.Module] = lambda c: nn.BatchNorm2d(c),
        dw_act: nn.Module | None = None,
        pw_act: nn.Module | None = None,
        pre_normact=False,
    ):
        pw_act = pw_act or self.default_pw_act
        super().__init__(
            DWConv3x3(
                c=c1,
                s=s,
                p=p,
                d=d,
                norm_fn=norm_fn,
                act=dw_act,
                pre_normact=pre_normact,
            ),
            Conv1x1(c1=c1, c2=c2, norm_fn=norm_fn, act=pw_act, pre_normact=pre_normact),
        )
