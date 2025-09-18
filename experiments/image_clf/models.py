from functools import partial

import torch
import torch.nn as nn

from legoml.nn.attention import SEAttention
from legoml.nn.contrib.convmixer import ConvMixerBlock, ConvMixerStem
from legoml.nn.contrib.convnext import (
    ConvNeXtBlock,
    ConvNeXtDownsample,
    ConvNextV2Block,
)
from legoml.nn.contrib.mobilenet import FusedMBConv, MBConv
from legoml.nn.contrib.resnet import (
    Res2NetBlock,
    Res2NetBottleneck,
    ResNetBlock,
    ResNetDShortcut,
    ResNetPreActBlock,
    ResNetPreActBlock_S2D_SE,
)
from legoml.nn.conv import Conv1x1, Conv3x3NormAct, NormActConv
from legoml.nn.mlp import FCNormAct
from legoml.nn.norm import GRN, LayerNorm2d
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
        blk = partial(
            ConvNextV2Block,
            block3=partial(
                NormActConv,
                k=1,
                # norm=partial(GRN, gamma_init=0.0, beta_init=0.0),
                norm=nn.BatchNorm2d,
                act=None,
            ),
        )
        self.backbone = nn.Sequential(
            blk(c_in=32, c_out=32),  # [32, 32, 32]
            blk(c_in=32, c_out=32),  # [32, 32, 32]
            ConvNeXtDownsample(c_in=32, c_out=64, s=2),  # [64, 16, 16]
            blk(c_in=64, c_out=64),  # [64, 16, 16]
            blk(c_in=64, c_out=64),  # [64, 16, 16]
            blk(c_in=64, c_out=64),  # [64, 16, 16]
            blk(c_in=64, c_out=64),  # [64, 16, 16]
            ConvNeXtDownsample(c_in=64, c_out=128, s=2),  # [128, 8, 8]
            blk(c_in=128, c_out=128),  # [128, 8, 8]
            blk(c_in=128, c_out=128),  # [128, 8, 8]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [128]
            nn.LayerNorm(128),
            nn.Linear(128, 10),
        )


class ConvNeXt_2x2_stem(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        self.stem = nn.Sequential(ConvMixerStem(c_in=c_in, c_out=64, k=2, s=2))
        blk = partial(
            ConvNextV2Block,
            block3=partial(
                NormActConv,
                k=1,
                # norm=partial(GRN, gamma_init=0.0, beta_init=0.0),
                norm=nn.BatchNorm2d,
                act=None,
            ),
        )
        self.backbone = nn.Sequential(
            blk(c_in=64, c_out=64),  # [64, 16, 16]
            blk(c_in=64, c_out=64),  # [64, 16, 16]
            blk(c_in=64, c_out=64),  # [64, 16, 16]
            blk(c_in=64, c_out=64),  # [64, 16, 16]
            blk(c_in=64, c_out=64),  # [64, 16, 16]
            blk(c_in=64, c_out=64),  # [64, 16, 16]
            ConvNeXtDownsample(c_in=64, c_out=128, s=2),  # [128, 8, 8]
            blk(c_in=128, c_out=128),  # [128, 8, 8]
            blk(c_in=128, c_out=128),  # [128, 8, 8]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [128]
            nn.LayerNorm(128),
            nn.Linear(128, 10),
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
            nn.Linear(1024, 10),
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


class ResNetPreActBlock_tiny_32x32(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        self.stem = nn.Sequential(
            Conv3x3NormAct(c_in=c_in, c_out=16),  # [16, 32, 32]
        )
        self.backbone = nn.Sequential(
            ResNetPreActBlock(c_in=16, c_out=16),  # [16, 32, 32]
            ResNetPreActBlock(c_in=16, c_out=16),  # [16, 32, 32]
            ResNetPreActBlock(c_in=16, c_out=16),  # [16, 32, 32]
            ResNetPreActBlock(c_in=16, c_out=32, s=2),  # [32, 16, 16]
            ResNetPreActBlock(c_in=32, c_out=32),  # [32, 16, 16]
            ResNetPreActBlock(c_in=32, c_out=32),  # [32, 16, 16]
            ResNetPreActBlock(c_in=32, c_out=64, s=2),  # [64, 8, 8]
            ResNetPreActBlock(c_in=64, c_out=64),  # [64, 8, 8]
            ResNetPreActBlock(c_in=64, c_out=64),  # [64, 8, 8]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [64]
            FCNormAct(c_in=64, c_out=10, act=nn.Identity),
        )


class ResNetWide_tiny_32x32(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        blk = partial(ResNetBlock, shortcut=ResNetDShortcut)
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


class ResNetPreActBlockWide_tiny_32x32(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        blk = ResNetPreActBlock_S2D_SE
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
            nn.Linear(128, 10),
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
            nn.Linear(256, 10),
        )


class ConvMixer_w256_d8_p2_k5(nn.Sequential):
    def __init__(self, c_in=3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvMixerStem(c_in=c_in, c_out=256, k=2, s=2),  # [256, 16, 16]
        )
        blk = partial(ConvMixerBlock, c_in=256, c_out=256, k=5)
        self.backbone = nn.Sequential(
            blk(),  # [256, 16, 16]
            blk(),  # [256, 16, 16]
            blk(),  # [256, 16, 16]
            blk(),  # [256, 16, 16]
            blk(),  # [256, 16, 16]
            blk(),  # [256, 16, 16]
            blk(),  # [256, 16, 16]
            blk(),  # [256, 16, 16]
        )
        self.head = nn.Sequential(
            GlobalAvgPool2d(),  # [256]
            nn.Linear(256, 10),
        )


if __name__ == "__main__":
    dummy_ip = torch.randn(1, 3, 32, 32)
    model = Res2Net_32x32()
    summarize_model(model, dummy_ip, depth=2)
