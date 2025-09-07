from functools import partial

import torch
import torch.nn as nn

from legoml.nn.conv import Conv1x1NormAct, Conv3x3NormAct
from legoml.nn.pool import GlobalAvgPool2d
from legoml.nn.types import ModuleCtor
from legoml.nn.utils import make_divisible


class SpatialAttention(nn.Module):
    def __init__(
        self,
        *,
        block: ModuleCtor = partial(
            Conv3x3NormAct,
            norm=None,
            act=nn.Sigmoid,
        ),
    ):
        super().__init__()
        self.block = block(c_in=2, c_out=1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        attn = self.block(torch.cat([avg_out, max_out], dim=1))  # [B, 1, H, W]
        return x * attn


class SEAttention(nn.Module):
    def __init__(
        self,
        *,
        c_in: int,
        c_mid: int | None = None,
        f_reduce: int = 4,
        pool: ModuleCtor = partial(GlobalAvgPool2d, keep_dim=True),
        block1: ModuleCtor = partial(Conv1x1NormAct, norm=None),
        block2: ModuleCtor = partial(Conv1x1NormAct, norm=None, act=nn.Sigmoid),
        act=partial(nn.ReLU, inplace=True),
    ):
        super().__init__()
        c_mid = c_mid or make_divisible(c_in / f_reduce)
        self.block = nn.Sequential(
            pool(),
            block1(c_in=c_in, c_out=c_mid),
            block2(c_in=c_mid, c_out=c_in),
        )

    def forward(self, x):
        return x * self.block(x)
