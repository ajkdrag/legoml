import torch.nn as nn

from legoml.nn.types import ModuleCtor
from legoml.nn.utils import autopad, identity


class PoolShortcut(nn.Sequential):
    def __init__(self, k: int = 3, s: int = 1, pool: ModuleCtor = nn.AvgPool2d):
        super().__init__()

        self.shortcut = (
            identity
            if s == 1
            else pool(
                kernel_size=k,
                stride=s,
                padding=autopad(k),
            )
        )
