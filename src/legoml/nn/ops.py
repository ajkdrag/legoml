import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class ChannelShuffle(nn.Sequential):
    """Channel shuffling for grouped convolution information exchange.

    Permutes channels to enable information flow between groups in grouped
    convolutions. Essential for maintaining representational capacity in
    efficient architectures like ShuffleNet.

    Parameters
    ----------
    g : int
        Number of groups for channel shuffling
    """

    def __init__(self, g):
        super().__init__()
        self.block = Rearrange("b (g c_per_g) h w -> b (c_per_g g) h w", g=g)


class LayerScale(nn.Module):
    def __init__(self, *, dims: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((1, dims, 1, 1), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class SpaceToDepth(nn.Sequential):
    def __init__(self, f_reduce: int):
        super().__init__()
        self.block = Rearrange(
            "b c (ho r1) (wo r2) -> b (c r1 r2) ho wo", r1=f_reduce, r2=f_reduce
        )
