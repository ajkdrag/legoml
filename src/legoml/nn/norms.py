import torch.nn as nn
from einops.layers.torch import Rearrange


class LayerNorm2d(nn.Sequential):
    def __init__(self, c: int):
        super().__init__(
            Rearrange("b c h w -> b h w c"),
            nn.LayerNorm(c),
            Rearrange("b h w c -> b c h w"),
        )


ln2d_fn = LayerNorm2d
ln_fn = nn.LayerNorm
bn1d_fn = nn.BatchNorm1d
bn2d_fn = nn.BatchNorm2d
