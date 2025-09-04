from typing import Callable

import torch.nn as nn

from legoml.nn.layers import (
    BranchAndConcat,
    Conv1x1NormAct,
    Conv3x3NormAct,
    ConvNormAct,
    DWConvNormAct,
    ScaledResidual,
    identity,
)
from legoml.nn.utils import make_divisible


class ConvNeXt(nn.Module):
    """ConvNeXt block with modernized ResNet design principles.

    Architecture: DW Conv (k×k, LN) -> 1×1 Expand (GELU) -> 1×1 Project
    Uses inverted bottleneck design with depthwise convolution, layer
    normalization instead of batch norm, larger kernels (7×7), and GELU
    activation. Applies residual connection with learnable scaling and
    optional stochastic depth.

    Parameters
    ----------
    c1 : int
        Input channels
    c2 : int, optional
        Output channels. Defaults to c1
    k : int, default=7
        Kernel size for depthwise convolution
    f_expand : int, default=2
        Expansion factor for intermediate channels
    s : int, default=1
        Stride
    p : int, optional
        Padding
    norm_fn : Callable, optional
        Normalization function. Defaults to ln2d_fn
    act_fn : Callable, optional
        Activation function. Defaults to gelu_fn
    drop_path : float, default=0.0
        Drop path probability for stochastic depth
    """

    def __init__(
        self,
        *,
        c1: int,
        c2: int | None = None,
        k: int = 7,
        f_expand: int = 2,
        s: int = 1,
        p: int | None = None,
        norm_fn: Callable[..., nn.Module] | None = None,
        act_fn: Callable[..., nn.Module] | None = None,
        drop_path: float = 0.0,
    ):
        super().__init__()
        c2 = c2 or c1
        c_ = make_divisible(c1 * f_expand)
        act_fn = act_fn or gelu_fn
        norm_fn = norm_fn or ln2d_fn

        self.dw_conv = DWConv(c=c1, k=k, s=s, act_fn=noop_fn, norm_fn=norm_fn)
        self.expand_conv = Conv1x1(c1=c1, c2=c_, act_fn=act_fn, norm_fn=noop_fn)
        self.project_conv = Conv1x1(c1=c_, c2=c2, act_fn=noop_fn, norm_fn=noop_fn)
        self.block = ScaledResidual(
            fn=nn.Sequential(self.dw_conv, self.expand_conv, self.project_conv),
            shortcut=Shortcut(c1=c1, c2=c2, s=s),
            drop_prob=drop_path,
        )

    def forward(self, x):
        return self.block(x)


class ResNetBasic(nn.Module):
    """Basic ResNet block with dual 3×3 convolution path.

    Architecture: 3×3 Conv (BN, ReLU) -> 3×3 Conv (BN) -> + Shortcut -> ReLU
    Enhanced with pre-activation support (BN->ReLU->Conv pattern), learnable
    residual scaling, and stochastic depth. In pre-activation mode, final
    activation occurs after residual addition. Shortcut uses identity when
    dimensions match, otherwise 1×1 projection (with BN in post-act mode).

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
        Padding
    norm_fn : Callable, optional
        Normalization function. Defaults to bn2d_fn
    act_fn : Callable, optional
        Activation function. Defaults to relu_fn
    pre_normact : bool, default=False
        If True, uses pre-activation (norm->act->conv) pattern
    drop_path : float, default=0.0
        Drop path probability for stochastic depth
    """

    def __init__(
        self,
        *,
        c1: int,
        c2: int | None = None,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        norm_fn: Callable[..., nn.Module] | None = None,
        act_fn: Callable[..., nn.Module] | None = None,
        pre_normact=False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        c2 = c2 or c1
        act_fn = act_fn or relu_fn
        norm_fn = norm_fn or bn2d_fn
        last_conv_act_fn: Callable[..., nn.Module] = act_fn if pre_normact else noop_fn
        post_res_act_fn: Callable[..., nn.Module] = noop_fn if pre_normact else act_fn
        shortcut_norm_fn: Callable[..., nn.Module] = noop_fn if pre_normact else norm_fn

        self.first_conv = ConvLayer(
            c1=c1,
            c2=c2,
            k=k,
            s=s,
            p=p,
            norm_fn=norm_fn,
            act_fn=act_fn,
            pre_normact=pre_normact,
        )
        self.second_conv = Conv3x3(
            c1=c2,
            c2=c2,
            norm_fn=norm_fn,
            act_fn=last_conv_act_fn,
            pre_normact=pre_normact,
        )
        self.shortcut = Shortcut(c1=c1, c2=c2, s=s, p=p, norm_fn=shortcut_norm_fn)
        self.scaled_residual = ScaledResidual(
            fn=nn.Sequential(self.first_conv, self.second_conv),
            shortcut=self.shortcut,
            drop_prob=drop_path,
        )
        self.post_res_act = post_res_act_fn()

    def forward(self, x):
        x = self.scaled_residual(x)
        return self.post_res_act(x)


class ResNetBottleneck(nn.Module):
    """Bottleneck ResNet block with efficient 1×1->3×3->1×1 design.

    Architecture: 1×1 Reduce -> 3×3 Conv -> 1×1 Expand -> + Shortcut
    Reduces computation by first compressing channels with 1×1 conv,
    applying expensive 3×3 spatial conv on reduced channels, then expanding
    back. Supports pre-activation variant where BN->ReLU precedes each conv.
    Enhanced with learnable scaling and stochastic depth regularization.

    Parameters
    ----------
    c1 : int
        Input channels
    c2 : int, optional
        Output channels. Defaults to c1
    k : int, default=3
        Kernel size for middle convolution
    f_reduce : int, default=2
        Channel reduction factor for bottleneck
    s : int, default=1
        Stride for middle convolution
    p : int, optional
        Padding for middle convolution
    norm_fn : Callable, optional
        Normalization function. Defaults to bn2d_fn
    act_fn : Callable, optional
        Activation function. Defaults to relu_fn
    pre_normact : bool, default=False
        If True, uses pre-activation pattern
    drop_path : float, default=0.0
        Drop path probability for stochastic depth
    """

    def __init__(
        self,
        *,
        c1: int,
        c2: int | None = None,
        k: int = 3,
        f_reduce: int = 2,
        s: int = 1,
        p: int | None = None,
        norm_fn: Callable[..., nn.Module] | None = None,
        act_fn: Callable[..., nn.Module] | None = None,
        pre_normact=False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        c2 = c2 or c1
        c_mid = make_divisible(c2 / f_reduce)
        act_fn = act_fn or relu_fn
        norm_fn = norm_fn or bn2d_fn
        last_conv_act_fn: Callable[..., nn.Module] = act_fn if pre_normact else noop_fn
        post_res_act_fn: Callable[..., nn.Module] = noop_fn if pre_normact else act_fn
        shortcut_norm_fn: Callable[..., nn.Module] = noop_fn if pre_normact else norm_fn

        self.reduce_conv = Conv1x1(
            c1=c1, c2=c_mid, norm_fn=norm_fn, act_fn=act_fn, pre_normact=pre_normact
        )
        self.middle_conv = ConvLayer(
            c1=c_mid,
            c2=c_mid,
            k=k,
            s=s,
            p=p,
            norm_fn=norm_fn,
            act_fn=act_fn,
            pre_normact=pre_normact,
        )
        self.expand_conv = Conv1x1(
            c1=c_mid,
            c2=c2,
            norm_fn=norm_fn,
            act_fn=last_conv_act_fn,
            pre_normact=pre_normact,
        )
        self.shortcut = Shortcut(c1=c1, c2=c2, s=s, p=p, norm_fn=shortcut_norm_fn)
        self.scaled_residual = ScaledResidual(
            fn=nn.Sequential(self.reduce_conv, self.middle_conv, self.expand_conv),
            shortcut=self.shortcut,
            drop_prob=drop_path,
        )
        self.post_res_act = post_res_act_fn()

    def forward(self, x):
        x = self.scaled_residual(x)
        return self.post_res_act(x)


class FusedMBConv(nn.Module):
    """Fused MobileNet block replacing DW+PW with single efficient convolution.

    Architecture: k×k Expand Conv (BN, SiLU) -> 1×1 Project (BN)
    Fuses the expansion and depthwise convolution steps of MBConv into a
    single k×k convolution for improved efficiency on hardware accelerators.
    Applies residual connection only when stride=1 and input/output channels
    match. No activation after final projection to preserve information flow.

    Parameters
    ----------
    c1 : int
        Input channels
    c2 : int, optional
        Output channels. Defaults to c1
    k : int, default=3
        Kernel size for expansion convolution
    f_expand : int, default=2
        Expansion factor for intermediate channels
    s : int, default=1
        Stride
    p : int, optional
        Padding
    norm_fn : Callable, optional
        Normalization function. Defaults to bn2d_fn
    act_fn : Callable, optional
        Activation function. Defaults to silu_fn
    pre_normact : bool, default=False
        If True, uses pre-activation pattern
    drop_path : float, default=0.0
        Drop path probability for stochastic depth
    """

    def __init__(
        self,
        *,
        c1: int,
        c2: int | None = None,
        k: int = 3,
        f_expand: int = 2,
        s: int = 1,
        p: int | None = None,
        norm_fn: Callable[..., nn.Module] | None = None,
        act_fn: Callable[..., nn.Module] | None = None,
        pre_normact: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        c2 = c2 or c1
        c_mid = make_divisible(c1 * f_expand)
        act_fn = act_fn or silu_fn
        norm_fn = norm_fn or bn2d_fn
        self.apply_residual = s == 1 and c1 == c2
        last_conv_act_fn: Callable[..., nn.Module] = act_fn if pre_normact else noop_fn

        self.expand_conv = ConvLayer(
            c1=c1,
            c2=c_mid,
            k=k,
            s=s,
            p=p,
            act_fn=act_fn,
            norm_fn=norm_fn,
            pre_normact=pre_normact,
        )
        self.project_conv = Conv1x1(
            c1=c_mid,
            c2=c2,
            act_fn=last_conv_act_fn,
            norm_fn=norm_fn,
            pre_normact=pre_normact,
        )
        if self.apply_residual:
            self.scaled_residual = ScaledResidual(
                fn=nn.Sequential(self.expand_conv, self.project_conv),
                drop_prob=drop_path,
            )

    def forward(self, x):
        if self.apply_residual:
            return self.scaled_residual(x)
        else:
            x = self.expand_conv(x)
            x = self.project_conv(x)
            return x


class MBConv(nn.Module):
    """MobileNetV2 inverted residual with depthwise separable convolution.

    Architecture: [1×1 Expand] -> k×k DW Conv (BN, ReLU6) -> 1×1 Project (BN)
    Inverted bottleneck design that expands channels, applies efficient
    depthwise convolution for spatial mixing, then projects back to output
    channels. Expansion step is skipped when input channels equal expanded
    channels. Residual connection only when stride=1 and c1=c2. Enhanced
    with learnable scaling and stochastic depth.

    Parameters
    ----------
    c1 : int
        Input channels
    c2 : int, optional
        Output channels. Defaults to c1
    k : int, default=3
        Kernel size for depthwise convolution
    f_expand : int, default=2
        Expansion factor for intermediate channels
    s : int, default=1
        Stride for depthwise convolution
    p : int, optional
        Padding for depthwise convolution
    norm_fn : Callable, optional
        Normalization function. Defaults to bn2d_fn
    act_fn : Callable, optional
        Activation function. Defaults to relu6_fn
    pre_normact : bool, default=False
        If True, uses pre-activation pattern
    drop_path : float, default=0.0
        Drop path probability for stochastic depth
    """

    def __init__(
        self,
        *,
        c1: int,
        c2: int | None = None,
        k: int = 3,
        f_expand: int = 2,
        s: int = 1,
        p: int | None = None,
        norm_fn: Callable[..., nn.Module] | None = None,
        act_fn: Callable[..., nn.Module] | None = None,
        pre_normact: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        c2 = c2 or c1
        c_mid = make_divisible(c1 * f_expand)
        self.use_expand = c_mid != c1
        act_fn = act_fn or relu6_fn
        norm_fn = norm_fn or bn2d_fn
        last_conv_act_fn: Callable[..., nn.Module] = act_fn if pre_normact else noop_fn
        self.apply_residual = s == 1 and c1 == c2

        self.expand_conv = (
            Conv1x1(
                c1=c1, c2=c_mid, norm_fn=norm_fn, act_fn=act_fn, pre_normact=pre_normact
            )
            if self.use_expand
            else identity
        )
        self.dw_conv = DWConv(
            c=c_mid,
            k=k,
            s=s,
            p=p,
            norm_fn=norm_fn,
            act_fn=act_fn,
            pre_normact=pre_normact,
        )
        self.project_conv = Conv1x1(
            c1=c_mid,
            c2=c2,
            norm_fn=norm_fn,
            act_fn=last_conv_act_fn,
            pre_normact=pre_normact,
        )

        if self.apply_residual:
            self.scaled_residual = ScaledResidual(
                fn=nn.Sequential(self.expand_conv, self.dw_conv, self.project_conv),
                drop_prob=drop_path,
            )

    def forward(self, x):
        if self.apply_residual:
            return self.scaled_residual(x)
        else:
            x = self.expand_conv(x)
            x = self.dw_conv(x)
            x = self.project_conv(x)
            return x


class ResNeXtBottleneck(nn.Module):
    """ResNeXt bottleneck with grouped convolutions for increased cardinality.

    Architecture: 1×1 Reduce -> 3×3 Grouped Conv -> 1×1 Expand -> + Shortcut
    Replaces standard 3×3 conv with grouped convolution to create multiple
    parallel paths (cardinality). Total intermediate channels = groups × width.
    This design increases model capacity while maintaining similar computational
    cost to standard bottleneck. Enhanced with learnable scaling and stochastic
    depth regularization.

    Parameters
    ----------
    c1 : int
        Input channels
    c2 : int, optional
        Output channels. Defaults to c1
    k : int, default=3
        Kernel size for grouped convolution
    g : int, default=16
        Number of groups for grouped convolution
    width : int, default=4
        Width (channels per group) for grouped convolution
    s : int, default=1
        Stride for grouped convolution
    p : int, optional
        Padding for grouped convolution
    norm_fn : Callable, optional
        Normalization function. Defaults to bn2d_fn
    act_fn : Callable, optional
        Activation function. Defaults to relu_fn
    pre_normact : bool, default=False
        If True, uses pre-activation pattern
    drop_path : float, default=0.0
        Drop path probability for stochastic depth
    """

    def __init__(
        self,
        *,
        c1: int,
        c2: int | None = None,
        k: int = 3,
        g: int = 16,
        width: int = 4,
        s: int = 1,
        p: int | None = None,
        norm_fn: Callable[..., nn.Module] | None = None,
        act_fn: Callable[..., nn.Module] | None = None,
        pre_normact=False,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        c2 = c2 or c1
        c_mid = g * width
        act_fn = act_fn or relu_fn
        norm_fn = norm_fn or bn2d_fn
        last_conv_act_fn: Callable[..., nn.Module] = act_fn if pre_normact else noop_fn
        post_res_act_fn: Callable[..., nn.Module] = noop_fn if pre_normact else act_fn
        shortcut_norm_fn: Callable[..., nn.Module] = noop_fn if pre_normact else norm_fn

        self.reduce_conv = Conv1x1(
            c1=c1, c2=c_mid, norm_fn=norm_fn, act_fn=act_fn, pre_normact=pre_normact
        )
        self.grouped_conv = ConvLayer(
            c1=c_mid,
            c2=c_mid,
            k=k,
            s=s,
            p=p,
            g=g,
            norm_fn=norm_fn,
            act_fn=act_fn,
            pre_normact=pre_normact,
        )
        self.expand_conv = Conv1x1(
            c1=c_mid,
            c2=c2,
            norm_fn=norm_fn,
            act_fn=last_conv_act_fn,
            pre_normact=pre_normact,
        )
        self.shortcut = Shortcut(c1=c1, c2=c2, s=s, p=p, norm_fn=shortcut_norm_fn)
        self.scaled_residual = ScaledResidual(
            fn=nn.Sequential(self.reduce_conv, self.grouped_conv, self.expand_conv),
            shortcut=self.shortcut,
            drop_prob=drop_path,
        )
        self.post_res_act = post_res_act_fn()

    def forward(self, x):
        x = self.scaled_residual(x)
        return self.post_res_act(x)


class InceptionResnet(nn.Module):
    """Inception-ResNet with multi-scale parallel branches and residual learning.

    Architecture: Branch1(1×1) || Branch2(1×1->3×3) || Branch3(1×1->3×3->3×3)
                 -> Concat -> 1×1 Project -> + Shortcut
    Three parallel branches capture features at different scales: direct 1×1,
    single 3×3, and cascaded 3×3 convolutions. All branches use 1×1 reduction
    to minimize parameters. Outputs are concatenated, projected to target
    channels, and added to shortcut with learnable scaling and stochastic depth.

    Parameters
    ----------
    c1 : int
        Input channels
    c2 : int, optional
        Output channels. Defaults to c1
    f_reduce : int, default=4
        Channel reduction factor for intermediate branches
    s : int, default=1
        Stride for 3x3 convolutions
    p : int, optional
        Padding for 3x3 convolutions
    norm_fn : Callable, optional
        Normalization function. Defaults to bn2d_fn
    act_fn : Callable, optional
        Activation function. Defaults to relu_fn
    pre_normact : bool, default=False
        If True, uses pre-activation pattern
    drop_path : float, default=0.0
        Drop path probability for stochastic depth
    """

    def __init__(
        self,
        *,
        c1: int,
        c2: int | None = None,
        f_reduce: int = 4,
        s: int = 1,
        p: int | None = None,
        norm_fn: Callable[..., nn.Module] | None = None,
        act_fn: Callable[..., nn.Module] | None = None,
        pre_normact: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        c2 = c2 or c1
        c_mid = make_divisible(c2 / f_reduce)
        act_fn = act_fn or relu_fn
        norm_fn = norm_fn or bn2d_fn
        last_conv_act_fn: Callable[..., nn.Module] = act_fn if pre_normact else noop_fn
        post_res_act_fn: Callable[..., nn.Module] = noop_fn if pre_normact else act_fn
        shortcut_norm_fn: Callable[..., nn.Module] = noop_fn if pre_normact else norm_fn

        self.branch1 = Conv1x1(
            c1=c1, c2=c_mid, norm_fn=norm_fn, act_fn=act_fn, pre_normact=pre_normact
        )

        self.branch2_reduce = Conv1x1(
            c1=c1, c2=c_mid, norm_fn=norm_fn, act_fn=act_fn, pre_normact=pre_normact
        )
        self.branch2_conv = Conv3x3(
            c1=c_mid,
            c2=c_mid,
            s=s,
            p=p,
            norm_fn=norm_fn,
            act_fn=act_fn,
            pre_normact=pre_normact,
        )

        self.branch3_reduce = Conv1x1(
            c1=c1, c2=c_mid, norm_fn=norm_fn, act_fn=act_fn, pre_normact=pre_normact
        )
        self.branch3_conv1 = Conv3x3(
            c1=c_mid,
            c2=c_mid,
            norm_fn=norm_fn,
            act_fn=act_fn,
            pre_normact=pre_normact,
        )
        self.branch3_conv2 = Conv3x3(
            c1=c_mid,
            c2=c_mid,
            norm_fn=norm_fn,
            act_fn=last_conv_act_fn,
            pre_normact=pre_normact,
        )

        self.branch_concat = BranchAndConcat(
            self.branch1,
            nn.Sequential(self.branch2_reduce, self.branch2_conv),
            nn.Sequential(self.branch3_reduce, self.branch3_conv1, self.branch3_conv2),
        )
        self.proj = Conv1x1(
            c1=c_mid * 3,
            c2=c2,
            norm_fn=norm_fn,
            act_fn=last_conv_act_fn,
            pre_normact=pre_normact,
        )
        self.shortcut = Shortcut(c1=c1, c2=c2, s=s, p=p, norm_fn=shortcut_norm_fn)
        self.scaled_residual = ScaledResidual(
            fn=nn.Sequential(self.branch_concat, self.proj),
            shortcut=self.shortcut,
            drop_prob=drop_path,
        )
        self.post_res_act = post_res_act_fn()

    def forward(self, x):
        x = self.scaled_residual(x)
        return self.post_res_act(x)
