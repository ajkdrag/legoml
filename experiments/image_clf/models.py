from functools import partial

import torch
import torch.nn as nn

from legoml.nn.attention import SEAttention
from legoml.nn.contrib.convnext import (
    ConvNeXtBlock,
    ConvNeXtDownsample,
    ConvNeXtDownsample_S2D,
    ConvNextV2Block,
)
from legoml.nn.contrib.mobilenet import FusedMBConv, MBConv
from legoml.nn.contrib.resnet import (
    Res2NetBlock,
    Res2NetBottleneck,
    ResNetBasic,
    ResNetDShortcut,
    ResNetPreAct,
    ResNetPreAct_D_SE,
    ResNetPreAct_S2D_SE,
)
from legoml.nn.conv import Conv1x1, Conv1x1NormAct, Conv3x3NormAct
from legoml.nn.mlp import FCNormAct
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
            FCNormAct(c_in=64, c_out=10, norm=None, act=None),
        )


class ConvNeXt_tiny_32x32(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        self.stem = nn.Sequential(
            Conv3x3NormAct(c_in=c_in, c_out=32),  # [32, 32, 32]
        )
        blk = ConvNextV2Block
        self.backbone = nn.Sequential(
            blk(c_in=32, c_out=32),  # [32, 32, 32]
            blk(c_in=32, c_out=32),  # [32, 32, 32]
            ConvNeXtDownsample_S2D(c_in=32, c_out=64, s=2),  # [64, 16, 16]
            blk(c_in=64, c_out=64),  # [64, 16, 16]
            blk(c_in=64, c_out=64),  # [64, 16, 16]
            blk(c_in=64, c_out=64),  # [64, 16, 16]
            blk(c_in=64, c_out=64),  # [64, 16, 16]
            ConvNeXtDownsample_S2D(c_in=64, c_out=128, s=2),  # [128, 8, 8]
            blk(c_in=128, c_out=128),  # [128, 8, 8]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [128]
            nn.LayerNorm(128),
            FCNormAct(c_in=128, c_out=10, norm=None, act=None),
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
            ConvNeXtDownsample(c_in=64, c_out=128, s=2),  # [128, 8, 8]
            ConvNeXtBlock(
                c_in=128,
                c_out=128,
                block3=Conv1x1SECtor(after=partial(SEAttention, c_in=128)),
            ),  # [128, 8, 8]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [128]
            nn.LayerNorm(128),
            FCNormAct(c_in=128, c_out=10, norm=None, act=None),
        )


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
            ResNetPreAct(c_in=16, c_out=32, s=2),  # [32, 16, 16]
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
        blk = partial(ResNetBasic, shortcut=ResNetDShortcut)
        self.stem = nn.Sequential(
            Conv3x3NormAct(c_in=c_in, c_out=64),  # [32, 32, 32]
        )
        self.backbone = nn.Sequential(
            blk(c_in=64, c_out=64),  # [32, 32, 32]
            blk(c_in=64, c_out=128, s=2),  # [64, 16, 16]
            blk(c_in=128, c_out=128),  # [64, 16, 16]
            blk(c_in=128, c_out=256, s=2),  # [128, 8, 8]
            blk(c_in=256, c_out=256),  # [128, 8, 8]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [128]
            FCNormAct(c_in=256, c_out=10, act=None, norm=None),
        )


class ResNetPreActWide_tiny_32x32(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        blk = ResNetPreAct_S2D_SE
        self.stem = nn.Sequential(
            Conv3x3NormAct(c_in=c_in, c_out=32),  # [32, 32, 32]
        )
        self.backbone = nn.Sequential(
            blk(c_in=32, c_out=32),  # [32, 32, 32]
            blk(c_in=32, c_out=64, s=2),  # [64, 16, 16]
            blk(c_in=64, c_out=64),  # [64, 16, 16]
            blk(c_in=64, c_out=128, s=2),  # [128, 8, 8]
            blk(c_in=128, c_out=128, act=nn.ReLU),  # [128, 8, 8]
        )
        self.head = nn.Sequential(
            nn.BatchNorm2d(128),
            GlobalAvgPool2d(),  # [128]
            FCNormAct(c_in=128, c_out=10, act=None, norm=None),
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
            FCNormAct(c_in=256, c_out=10, norm=None, act=None),
        )


if __name__ == "__main__":
    dummy_ip = torch.randn(1, 3, 32, 32)
    model = ResNetPreActWide_tiny_32x32()
    summarize_model(model, dummy_ip, depth=2)
