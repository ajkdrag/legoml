import torch
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


class GRN(nn.Module):
    def __init__(
        self,
        dims: int,
        *,
        gamma_init: float = 0.1,
        beta_init: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.full((1, dims, 1, 1), gamma_init))
        self.beta = nn.Parameter(torch.full((1, dims, 1, 1), beta_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)  # [B, C, 1, 1]
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)  # [B, C, 1, 1]
        return x + self.gamma * (x * nx) + self.beta
