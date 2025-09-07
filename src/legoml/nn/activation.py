import torch.nn as nn

# class LayerNorm2d(nn.LayerNorm):
#     def forward(self, input):
#         # [B, C, H, W] -> [B, H, W, C]
#         input = input.permute(0, 2, 3, 1)
#         input = super().forward(input)
#         input = input.permute(0, 3, 1, 2)
#         return input
#


class LayerNorm2d(nn.GroupNorm):
    def __init__(self, dims: int):
        super().__init__(num_groups=1, num_channels=dims)


