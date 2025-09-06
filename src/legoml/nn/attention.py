from functools import partial

import torch
import torch.nn as nn

from legoml.nn.conv import Conv3x3NormAct


class SpatialAttention(nn.Module):
    def __init__(self, block=partial(Conv3x3NormAct, norm=None, act=nn.Sigmoid)):
        super().__init__()
        self.block = block(c_in=2, c_out=1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        attn = self.block(torch.cat([avg_out, max_out], dim=1))  # [B, 1, H, W]
        return x * attn
