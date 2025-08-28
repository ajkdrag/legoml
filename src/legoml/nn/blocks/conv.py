from typing import Callable
import torch
import torch.nn as nn

from legoml.nn.layers import (
    Conv1x1,
    Conv3x3,
    DWConv3x3,
    BranchAndConcat,
    ScaledResidual,
    Shortcut,
    DropPath,
    identity,
)


class ResNetBottleneck(nn.Sequential):
    """ResNet bottleneck block with 1x1 -> 3x3 -> 1x1 conv sequence
    Economizes parameters while maintaining capacity.
    """

    default_act = nn.ReLU(inplace=True)

    def __init__(
        self,
        *,
        c1: int,
        c2: int | None = None,
        f: int = 2,
        s: int = 1,
        p: int | None = None,
        norm_fn: Callable[..., nn.Module] = lambda c: nn.BatchNorm2d(c),
        act: nn.Module | None = None,
        pre_normact=False,
        drop_path: float = 0.0,
    ):
        c2 = c2 or c1
        c_mid = max(1, c2 // f)  # bottleneck compression (e.g., 4×)
        use_bottleneck = c_mid != c2
        act = act or self.default_act

        last_conv_act = act if pre_normact else identity
        residual_post_act = identity if pre_normact else act

        def shortcut_norm(c: int):
            return identity if pre_normact else norm_fn(c)

        main = nn.Sequential(
            Conv1x1(c1=c1, c2=c_mid, norm_fn=norm_fn, act=act, pre_normact=pre_normact),
            Conv3x3(
                c1=c_mid,
                c2=c_mid,
                s=s,
                p=p,
                norm_fn=norm_fn,
                act=act,
                pre_normact=pre_normact,
            ),
            Conv1x1(
                c1=c_mid,
                c2=c2,
                norm_fn=norm_fn,
                act=last_conv_act,
                pre_normact=pre_normact,
            )
            if use_bottleneck
            else identity,
            DropPath(drop_path),
        )

        shortcut = Shortcut(c1=c1, c2=c2, s=s, p=p, norm_fn=shortcut_norm)

        super().__init__(
            ScaledResidual(fn=main, shortcut=shortcut),
            residual_post_act,
        )


class MBConv(nn.Sequential):
    """
    MobileNetV2 Inverted Residual:
      [optional] 1x1 expand  (BN+Act if post-act; BN->Act->Conv if pre-act)
                 3x3 depthwise (stride s)
                 1x1 project (no nonlinearity).

    Residual only when s==1 and c1==c2.
    """

    default_act = nn.ReLU6(inplace=True)

    def __init__(
        self,
        *,
        c1: int,
        c2: int | None = None,
        f: int = 2,
        s: int = 1,
        p: int | None = None,
        norm_fn: Callable[[int], nn.Module] = lambda c: nn.BatchNorm2d(c),
        act: nn.Module | None = None,
        pre_normact: bool = False,
        drop_path: float = 0.0,
    ):
        c2 = c2 or c1
        c_mid = max(1, int(round(c1 * f)))
        use_expand = c_mid != c1
        act = act or self.default_act

        last_conv_act = act if pre_normact else identity
        apply_residual = s == 1 and c1 == c2

        layers: list[nn.Module] = [
            Conv1x1(c1=c1, c2=c_mid, norm_fn=norm_fn, act=act, pre_normact=pre_normact)
            if use_expand
            else identity,
            DWConv3x3(
                c=c_mid, s=s, p=p, norm_fn=norm_fn, act=act, pre_normact=pre_normact
            ),
            Conv1x1(
                c1=c_mid,
                c2=c2,
                norm_fn=norm_fn,
                act=last_conv_act,
                pre_normact=pre_normact,
            ),
            DropPath(drop_path),
        ]

        if apply_residual:
            main = nn.Sequential(*layers)
            layers = [ScaledResidual(fn=main)]
        super().__init__(*layers)


class ResNeXtBottleneck(nn.Sequential):
    """Bottleneck block with grouped convolution.
    (c, h, w) -> `g` branches of (1x1, 3x3, 1x1) with each branch having
    `c_mid=4` (width).
    This can be "simplified" using grouped convs where `c_mid=g*4`.
    """

    default_act = nn.ReLU(inplace=True)

    def __init__(
        self,
        *,
        c1: int,
        c2: int | None = None,
        g: int = 16,
        width: int = 4,
        s: int = 1,
        p: int | None = None,
        norm_fn: Callable[..., nn.Module] = lambda c: nn.BatchNorm2d(c),
        act: nn.Module | None = None,
        pre_normact=False,
        drop_path: float = 0.0,
    ) -> None:
        c2 = c2 or c1
        c_mid = g * width
        use_bottleneck = c_mid != c2
        act = act or self.default_act

        last_conv_act = act if pre_normact else identity
        residual_post_act = identity if pre_normact else act

        def shortcut_norm(c: int):
            return identity if pre_normact else norm_fn(c)

        main = nn.Sequential(
            Conv1x1(c1=c1, c2=c_mid, norm_fn=norm_fn, act=act, pre_normact=pre_normact),
            Conv3x3(
                c1=c_mid,
                c2=c_mid,
                s=s,
                p=p,
                g=g,
                norm_fn=norm_fn,
                act=act,
                pre_normact=pre_normact,
            ),
            Conv1x1(
                c1=c_mid,
                c2=c2,
                norm_fn=norm_fn,
                act=last_conv_act,
                pre_normact=pre_normact,
            )
            if use_bottleneck
            else identity,
            DropPath(drop_path),
        )

        shortcut = Shortcut(c1=c1, c2=c2, s=s, p=p, norm_fn=shortcut_norm)
        super().__init__(
            ScaledResidual(fn=main, shortcut=shortcut),
            residual_post_act,
        )


# class InceptionResnet(nn.Sequential):
#     """Simplified Inception architecture with residual connections
#     c1 -> c2 via (nxn) branch is replaced by [(c1 -> c_mid) via (1x1)
#     followed by (c_mid -> c2) via (nxn)]:
#
#     #params before = (n x n x c1 x c2)
#     #params after = (1 x 1 x c1 x c_mid) + (n x n x c_mid x c2)
#
#     For n=3, and c_mid = c1 // 4, #params is reduced significantly
#     """
#
#     def __init__(
#         self,
#         *,
#         c1,
#         c2=None,
#         f=4,
#         act_fn=lambda: nn.ReLU(inplace=True),
#     ):
#         c2 = c2 or c1
#         c_mid = c1 // f
#         b1 = Conv_1x1__BnAct(c1=c1, c2=c_mid)
#         b2 = nn.Sequential(
#             Conv_1x1__BnAct(c1=c1, c2=c_mid),
#             Conv_3x3__BnAct(c1=c_mid, c2=c_mid),
#         )
#         b3 = nn.Sequential(
#             Conv_1x1__BnAct(c1=c1, c2=c_mid),
#             Conv_3x3__BnAct(c1=c_mid, c2=c_mid),
#             Conv_3x3__BnAct(c1=c_mid, c2=c_mid),
#         )
#
#         proj = nn.Conv2d(c_mid * 3, c2, kernel_size=1)
#         shortcut = Shortcut(c1=c1, c2=c2)
#         super().__init__(
#             ScaledResidual(BranchAndConcat(b1, b2, b3), proj, shortcut=shortcut),
#             act_fn(),
#         )
