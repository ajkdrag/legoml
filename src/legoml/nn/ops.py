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
    """
    Dimension-agnostic LayerScale.
    - If input is (N, C, H, W): gamma -> (1, C, 1, 1)
    - If input is (N, D):       gamma -> (1, D)
    - If input is (N, L, D):    gamma -> (1, 1, D)
    """

    def __init__(self, *, dims: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dims), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:  # (N, C, H, W) -> CNNs
            shape = (1, -1, 1, 1)
        elif x.dim() == 3:  # (N, L, D) -> Transformers
            shape = (1, 1, -1)
        elif x.dim() == 2:  # (N, D) -> MLPs
            shape = (1, -1)
        else:
            raise ValueError(f"Unsupported input rank {x.dim()} for LayerScaler")

        gamma = self.gamma.view(*shape)
        return x * gamma
