import torch.nn as nn

from legoml.nn.utils import identity


class GlobalAvgPool2d(nn.Sequential):
    """Global average pooling for spatial dimension reduction.

    Reduces spatial dimensions (H, W) to (1, 1) via adaptive average pooling.
    Commonly used as final pooling layer in CNNs before classification head.

    Parameters
    ----------
    keep_dim : bool, default=False
        If True, keeps spatial dimensions as (1, 1). If False, flattens to 1D
    """

    def __init__(self, keep_dim: bool = False):
        super().__init__()
        self.block = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten() if not keep_dim else identity
