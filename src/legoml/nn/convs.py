import torch
import torch.nn as nn

from .common import (
    Conv_3x3_Down__BnAct,
    Conv_1x1__BnAct,
    Conv_3x3__BnAct,
    Conv__BnAct,
    DWConv_3x3__BnAct,
    DWConv_3x3_Down__BnAct,
    Shortcut,
    Shortcut_Down,
)


class ResBottleneck(nn.Module):
    """ResNet bottleneck block with 1x1 -> 3x3 -> 1x1 conv sequence"""

    def __init__(self, *, c1, c2, f=4, act_fn=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        c_mid = c2 // f  # bottleneck compression (e.g., 4×)
        self.conv1 = Conv_1x1__BnAct(c1=c1, c2=c_mid)
        self.conv2 = Conv_3x3__BnAct(c1=c_mid, c2=c_mid)
        self.conv3 = Conv_1x1__BnAct(c1=c_mid, c2=c2, act_fn=lambda: nn.Identity())
        self.shortcut = Shortcut(c1=c1, c2=c2)
        self.act = act_fn()

    def forward(self, x):
        residual = self.shortcut(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        return self.act(y + residual)


class ResBottleneck_Down(nn.Module):
    """ResNet bottleneck block (1x1 -> 3x3 stride 2 -> 1x1) but for downscaling by 2"""

    def __init__(self, *, c1, c2, f=4, act_fn=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        c_mid = c2 // f  # bottleneck compression (e.g., 4×)
        self.conv1 = Conv_1x1__BnAct(c1=c1, c2=c_mid)
        self.conv2 = Conv_3x3_Down__BnAct(c1=c_mid, c2=c_mid)
        self.conv3 = Conv_1x1__BnAct(c1=c_mid, c2=c2, act_fn=lambda: nn.Identity())
        self.shortcut = Shortcut_Down(c1=c1, c2=c2)
        self.act = act_fn()

    def forward(self, x):
        residual = self.shortcut(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        return self.act(y + residual)


class MBConv(nn.Module):
    """MobileNetV2 Inverted Residual Block, 1x1 expand -> depthwise 3x3 -> 1x1 project.
    Residual connection is applied only if output channels match input
    """

    def __init__(self, *, c1, c2, f=6, act_fn=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        c_mid = c1 * f
        self.use_shortcut = c1 == c2
        self.expand_conv = Conv_1x1__BnAct(
            c1=c1,
            c2=c_mid,
            act_fn=lambda: nn.ReLU6(inplace=True),
        )
        self.dw_conv = DWConv_3x3__BnAct(
            c1=c_mid, act_fn=lambda: nn.ReLU6(inplace=True)
        )
        self.project_conv = Conv_1x1__BnAct(
            c1=c_mid,
            c2=c2,
            act_fn=lambda: nn.Identity(),
        )
        self.act = act_fn()

    def forward(self, x):
        y = self.expand_conv(x)
        y = self.dw_conv(y)
        y = self.project_conv(y)
        return self.act(x + y if self.use_shortcut else y)


class MBConv_Down(nn.Module):
    """MobileNetV2 Inverted Residual Block, 1x1 expand -> depthwise 3x3 -> 1x1 project.
    Residual connection is not applied and downscale by 2
    """

    def __init__(self, *, c1, c2, f=6, act_fn=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        c_mid = c1 * f
        self.expand_conv = Conv_1x1__BnAct(
            c1=c1,
            c2=c_mid,
            act_fn=lambda: nn.ReLU6(inplace=True),
        )
        self.dw_conv = DWConv_3x3_Down__BnAct(
            c1=c_mid, act_fn=lambda: nn.ReLU6(inplace=True)
        )
        self.project_conv = Conv_1x1__BnAct(
            c1=c_mid,
            c2=c2,
            act_fn=lambda: nn.Identity(),
        )
        self.act = act_fn()

    def forward(self, x):
        y = self.expand_conv(x)
        y = self.dw_conv(y)
        y = self.project_conv(y)
        return self.act(y)


class ResNeXtBottleneck(nn.Module):
    """Bottleneck block with grouped convolution.
    (c, h, w) -> `g` branches of (1x1, 3x3, 1x1) with each branch having `c_mid=4` (width).
    This can be "simplified" using grouped convs where `c_mid=g*4`
    """

    def __init__(
        self, *, c1, c2, g, width=4, act_fn=lambda: nn.ReLU(inplace=True)
    ) -> None:
        super().__init__()
        c_mid = g * width
        self.conv_in = Conv_1x1__BnAct(c1=c1, c2=c_mid)
        self.conv_group = Conv__BnAct(c1=c_mid, c2=c_mid, g=g)
        self.conv_out = Conv_1x1__BnAct(c1=c_mid, c2=c2)
        self.shortcut = Shortcut(c1=c1, c2=c2)
        self.act = act_fn()

    def forward(self, x):
        res = self.shortcut(x)
        y = self.conv_in(x)
        y = self.conv_group(y)
        y = self.conv_out(y)
        return self.act(res + y)


class ResNeXtBottleneck_Down(nn.Module):
    def __init__(
        self, *, c1, c2, g, width=4, act_fn=lambda: nn.ReLU(inplace=True)
    ) -> None:
        super().__init__()
        c_mid = g * width
        self.conv_in = Conv_1x1__BnAct(c1=c1, c2=c_mid)
        self.conv_group = Conv__BnAct(c1=c_mid, c2=c_mid, g=g, s=2)
        self.conv_out = Conv_1x1__BnAct(c1=c_mid, c2=c2)
        self.shortcut = Shortcut_Down(c1=c1, c2=c2)
        self.act = act_fn()

    def forward(self, x):
        res = self.shortcut(x)
        y = self.conv_in(x)
        y = self.conv_group(y)
        y = self.conv_out(y)
        return self.act(res + y)


class InceptionResnet(nn.Module):
    """Simplified Inception architecture with residual connections
    c1 -> c2 via (nxn) branch is replaced by [(c1 -> c_mid) via (1x1)
    followed by (c_mid -> c2) via (nxn)]:

    #params before = (n x n x c1 x c2)
    #params after = (1 x 1 x c1 x c_mid) + (n x n x c_mid x c2)

    For n=3, and c_mid = c1 // 4, #params is reduced significantly
    """

    def __init__(
        self,
        *,
        c1,
        c2=None,
        f=4,
        act_fn=lambda: nn.ReLU(inplace=True),
    ):
        super().__init__()
        c2 = c2 or c1
        c_mid = c1 // f
        self.b1 = Conv_1x1__BnAct(c1=c1, c2=c_mid)
        self.b2 = nn.Sequential(
            Conv_1x1__BnAct(c1=c1, c2=c_mid),
            Conv_3x3__BnAct(c1=c_mid, c2=c_mid),
        )
        self.b3 = nn.Sequential(
            Conv_1x1__BnAct(c1=c1, c2=c_mid),
            Conv_3x3__BnAct(c1=c_mid, c2=c_mid),
            Conv_3x3__BnAct(c1=c_mid, c2=c_mid),
        )

        self.proj = nn.Conv2d(c_mid * 3, c2, kernel_size=1)
        self.shortcut = Shortcut(c1=c1, c2=c2)
        self.act = act_fn()

    def forward(self, x):
        res = self.shortcut(x)
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        y = self.proj(torch.cat([b1, b2, b3], dim=1))
        return self.act(res + y)
