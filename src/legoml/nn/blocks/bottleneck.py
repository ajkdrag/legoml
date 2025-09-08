from functools import partial

import torch.nn as nn

from legoml.nn.struct import ResidualAdd
from legoml.nn.types import ModuleCtor
from legoml.nn.utils import make_divisible


class Bottleneck(nn.Sequential):
    def __init__(
        self,
        *,
        c_in: int,
        c_mid: int | None = None,
        c_out: int | None = None,
        s: int = 1,
        f_reduce: int = 4,
        block1: ModuleCtor,
        block2: ModuleCtor,
        block3: ModuleCtor,
        shortcut: ModuleCtor,
        residual: ModuleCtor = ResidualAdd,
        act: ModuleCtor = partial(nn.ReLU, inplace=True),
    ):
        super().__init__()

        c_out = c_out or c_in
        c_mid = c_mid or make_divisible(c_out / f_reduce)

        block1 = block1(c_in=c_in, c_out=c_mid, s=1)
        block2 = block2(c_in=c_mid, c_out=c_mid, s=s)  # ResNet-D style stride
        block3 = block3(c_in=c_mid, c_out=c_out, s=1)
        shortcut = shortcut(c_in=c_in, c_out=c_out, s=s)

        self.block = residual(
            block=nn.Sequential(block1, block2, block3),
            shortcut=shortcut,
        )
        self.act = act()
