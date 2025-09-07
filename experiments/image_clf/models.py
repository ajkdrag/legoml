from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from legoml.nn.attention import SEAttention
from legoml.nn.blocks.convnext import ConvNeXtBlock, ConvNeXtDownsample
from legoml.nn.blocks.mobilenet import FusedMBConv, MBConv
from legoml.nn.blocks.resnet import (
    Res2NetBlock,
    Res2NetBottleneck,
    ResNetBasic,
    ResNetPreAct,
)
from legoml.nn.conv import Conv1x1, Conv3x3NormAct
from legoml.nn.mlp import FCNormAct
from legoml.nn.ops import LayerScale
from legoml.nn.pool import GlobalAvgPool2d
from legoml.nn.struct import ApplyAfterCtor
from legoml.utils.summary import summarize_model


class CNN__MLP_tiny_32x32(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        self.stem = nn.Sequential(
            Conv3x3NormAct(c_in=c_in, c_out=32),  # [32, 32, 32]
            nn.MaxPool2d(2, 2),  # [32, 16, 16]
        )
        self.backbone = nn.Sequential(
            Conv3x3NormAct(c_in=32, c_out=64),  # [64, 16, 16]
            nn.MaxPool2d(2, 2),  # [64, 8, 8]
            Conv3x3NormAct(c_in=64, c_out=64),  # [64, 8, 8]
            Conv3x3NormAct(c_in=64, c_out=64),  # [64, 8, 8]
            Conv3x3NormAct(c_in=64, c_out=64),  # [64, 8, 8]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [64]
            FCNormAct(c_in=64, c_out=10, act=nn.Identity),
        )


class ConvNeXt_tiny_32x32(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        self.stem = nn.Sequential(
            Conv3x3NormAct(c_in=c_in, c_out=32),  # [32, 32, 32]
        )
        self.backbone = nn.Sequential(
            ConvNeXtBlock(c_in=32, c_out=32),  # [32, 32, 32]
            ConvNeXtBlock(c_in=32, c_out=32),  # [32, 32, 32]
            ConvNeXtDownsample(c_in=32, c_out=64, s=2),  # [64, 16, 16]
            ConvNeXtBlock(c_in=64, c_out=64),  # [64, 16, 16]
            ConvNeXtBlock(c_in=64, c_out=64),  # [64, 16, 16]
            ConvNeXtBlock(c_in=64, c_out=64),  # [64, 16, 16]
            ConvNeXtBlock(c_in=64, c_out=64),  # [64, 16, 16]
            ConvNeXtDownsample(c_in=64, c_out=128, s=2),  # [128, 8, 8]
            ConvNeXtBlock(c_in=128, c_out=128),  # [128, 8, 8]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [96]
            FCNormAct(c_in=128, c_out=10, act=nn.Identity),
        )


class ConvNeXt_SE_32x32(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        Conv1x1SECtor = partial(ApplyAfterCtor, main=Conv1x1)

        self.stem = nn.Sequential(
            Conv3x3NormAct(c_in=c_in, c_out=32),  # [32, 32, 32]
        )
        self.backbone = nn.Sequential(
            ConvNeXtBlock(
                c_in=32,
                c_out=32,
                block3=Conv1x1SECtor(after=partial(SEAttention, c_in=32)),
            ),  # [32, 32, 32]
            ConvNeXtBlock(
                c_in=32,
                c_out=32,
                block3=Conv1x1SECtor(after=partial(SEAttention, c_in=32)),
            ),  # [32, 32, 32]
            ConvNeXtDownsample(c_in=32, c_out=64, s=2),  # [64, 16, 16]
            ConvNeXtBlock(
                c_in=64,
                c_out=64,
                block3=Conv1x1SECtor(after=partial(SEAttention, c_in=64)),
            ),  # [64, 16, 16]
            ConvNeXtBlock(
                c_in=64,
                c_out=64,
                block3=Conv1x1SECtor(after=partial(SEAttention, c_in=64)),
            ),  # [64, 16, 16]
            ConvNeXtBlock(
                c_in=64,
                c_out=64,
                block3=Conv1x1SECtor(after=partial(SEAttention, c_in=64)),
            ),  # [64, 16, 16]
            ConvNeXtBlock(
                c_in=64,
                c_out=64,
                block3=Conv1x1SECtor(after=partial(SEAttention, c_in=64)),
            ),  # [64, 16, 16]
            ConvNeXtBlock(
                c_in=64,
                c_out=64,
                block3=Conv1x1SECtor(after=partial(SEAttention, c_in=64)),
            ),  # [64, 16, 16]
            ConvNeXtDownsample(c_in=64, c_out=128, s=2),  # [128, 8, 8]
            ConvNeXtBlock(
                c_in=128,
                c_out=128,
                block3=Conv1x1SECtor(after=partial(SEAttention, c_in=128)),
            ),  # [128, 8, 8]
            ConvNeXtBlock(
                c_in=128,
                c_out=128,
                block3=Conv1x1SECtor(after=partial(SEAttention, c_in=128)),
            ),  # [128, 8, 8]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [128]
            nn.LayerNorm(128),
            FCNormAct(
                c_in=128,
                c_out=10,
                norm=None,
                act=None,
            ),
        )

    def param_groups(model):
        decay, no_decay = [], []
        for module in model.modules():
            if isinstance(
                module,
                (
                    nn.LayerNorm,
                    nn.GroupNorm,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    LayerScale,
                ),
            ):
                for p in module.parameters(recurse=False):
                    if p.requires_grad:
                        no_decay.append(p)
            else:
                for name, p in module.named_parameters(recurse=False):
                    if not p.requires_grad:
                        continue
                    (no_decay if name.endswith("bias") else decay).append(p)
        # sanity check
        assert len(decay) + len(no_decay) == len(list(model.parameters()))

        return decay, no_decay


class Res2Net_32x32(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        self.stem = nn.Sequential(
            Conv3x3NormAct(c_in=c_in, c_out=64),  # [32, 32, 32]
        )
        self.backbone = nn.Sequential(
            Res2NetBottleneck(
                c_in=64,
                block2=partial(Res2NetBlock, c_mid=32),
                c_out=256,
            ),  # [128, 32, 32]
            Res2NetBlock(c_in=256, c_mid=32, c_out=256),  # [128, 32, 32]
            Res2NetBottleneck(
                c_in=256,
                block2=partial(Res2NetBlock, c_mid=32),
                c_out=512,
                s=2,
            ),  # [256, 16, 16]
            Res2NetBlock(c_in=512, c_mid=64, c_out=512),  # [512, 8, 8]
            Res2NetBottleneck(
                c_in=512,
                block2=partial(Res2NetBlock, c_mid=32),
                c_out=1024,
                s=2,
            ),  # [512, 8, 8]
            Res2NetBlock(c_in=1024, c_mid=128, c_out=1024),  # [1024, 8, 8]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [1024]
            FCNormAct(c_in=1024, c_out=10, act=nn.Identity),
        )


class Res2NetWide_32x32(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        self.stem = nn.Sequential(
            Conv3x3NormAct(c_in=c_in, c_out=64),  # [64, 32, 32]
        )
        self.backbone = nn.Sequential(
            Res2NetBlock(c_in=64, c_mid=32),  # [64, 32, 32]
            Res2NetBlock(c_in=64, c_mid=32),  # [64, 32, 32]
            Res2NetBlock(c_in=64, c_mid=32, c_out=128, s=2),  # [128, 16, 16]
            Res2NetBlock(c_in=128, c_mid=32),  # [128, 16, 16]
            Res2NetBlock(c_in=128, c_mid=32),  # [128, 16, 16]
            Res2NetBlock(c_in=128, c_mid=32, c_out=256, s=2),  # [256, 8, 8]
            Res2NetBlock(c_in=256, c_mid=32),  # [256, 8, 8]
            Res2NetBlock(c_in=256, c_mid=32),  # [256, 8, 8]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [256]
            FCNormAct(c_in=256, c_out=10, act=nn.Identity),
        )


class ResNetPreAct_tiny_32x32(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        self.stem = nn.Sequential(
            Conv3x3NormAct(c_in=c_in, c_out=16),  # [16, 32, 32]
        )
        self.backbone = nn.Sequential(
            ResNetPreAct(c_in=16, c_out=16),  # [16, 32, 32]
            ResNetPreAct(c_in=16, c_out=16),  # [16, 32, 32]
            ResNetPreAct(c_in=16, c_out=16),  # [16, 32, 32]
            ResNetPreAct(c_in=16, c_out=32, s=2, drop_path=0.2),  # [32, 16, 16]
            ResNetPreAct(c_in=32, c_out=32),  # [32, 16, 16]
            ResNetPreAct(c_in=32, c_out=32),  # [32, 16, 16]
            ResNetPreAct(c_in=32, c_out=64, s=2),  # [64, 8, 8]
            ResNetPreAct(c_in=64, c_out=64),  # [64, 8, 8]
            ResNetPreAct(c_in=64, c_out=64),  # [64, 8, 8]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [64]
            FCNormAct(c_in=64, c_out=10, act=nn.Identity),
        )


class ResNetWide_tiny_32x32(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        self.stem = nn.Sequential(
            Conv3x3NormAct(c_in=c_in, c_out=16),  # [16, 32, 32]
        )
        self.backbone = nn.Sequential(
            ResNetBasic(c_in=16, c_out=16),  # [16, 32, 32]
            ResNetBasic(c_in=16, c_out=48, s=2, drop_path=0.1),  # [96, 16, 16]
            ResNetBasic(c_in=48, c_out=48),  # [96, 16, 16]
            ResNetBasic(c_in=48, c_out=96, s=2),  # [96, 8, 8]
            ResNetBasic(c_in=96, c_out=96),  # [96, 8, 8]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [96]
            FCNormAct(c_in=96, c_out=10, act=nn.Identity),
        )


class ResNetBasic_tiny_32x32(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        self.stem = nn.Sequential(
            Conv3x3NormAct(c_in=c_in, c_out=16),  # [16, 32, 32]
        )
        self.backbone = nn.Sequential(
            ResNetBasic(c_in=16, c_out=16),  # [16, 32, 32]
            ResNetBasic(c_in=16, c_out=16),  # [16, 32, 32]
            ResNetBasic(c_in=16, c_out=16),  # [16, 32, 32]
            ResNetBasic(c_in=16, c_out=32, s=2, drop_path=0.2),  # [32, 16, 16]
            ResNetBasic(c_in=32, c_out=32),  # [32, 16, 16]
            ResNetBasic(c_in=32, c_out=32),  # [32, 16, 16]
            ResNetBasic(c_in=32, c_out=64, s=2),  # [64, 8, 8]
            ResNetBasic(c_in=64, c_out=64),  # [64, 8, 8]
            ResNetBasic(c_in=64, c_out=64),  # [64, 8, 8]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [64]
            FCNormAct(c_in=64, c_out=10, act=nn.Identity),
        )


class MobileNet_tiny_32x32(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        self.stem = nn.Sequential(
            Conv3x3NormAct(c_in=c_in, c_out=64),  # [64, 32, 32]
        )
        self.backbone = nn.Sequential(
            FusedMBConv(c_in=64, c_out=64),  # [64, 32, 32]
            FusedMBConv(c_in=64, c_out=128, s=2),  # [128, 16, 16]
            FusedMBConv(c_in=128, c_out=128),  # [128, 16, 16]
            MBConv(c_in=128, c_out=256, s=2),  # [256, 8, 8]
            MBConv(c_in=256, c_out=256),  # [256, 8, 8]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [96]
            FCNormAct(c_in=256, c_out=10, act=nn.Identity),
        )


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    ):
        super().__init__()

        # stem and 3 intermediate downsampling conv layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(len(dims) - 1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # global average pooling, (N, C, H, W) -> (N, C)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def convnext_tiny(**kwargs):
    model = ConvNeXt(depths=[3, 6, 3], dims=[64, 128, 256], num_classes=10, **kwargs)
    return model


if __name__ == "__main__":
    dummy_ip = torch.randn(1, 3, 32, 32)
    model = convnext_tiny()
    summarize_model(model, dummy_ip, depth=2)
