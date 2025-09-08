import torch.nn as nn


class LegacyLayerNorm2d(nn.LayerNorm):
    def forward(self, input):
        # [B, C, H, W] -> [B, H, W, C]
        input = input.transpose(1, -1)
        input = super().forward(input)
        input = input.transpose(1, -1)
        return input


class LayerNorm2d(nn.GroupNorm):
    def __init__(self, dims: int):
        super().__init__(num_groups=1, num_channels=dims)
