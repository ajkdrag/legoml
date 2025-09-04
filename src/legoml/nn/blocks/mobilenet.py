from functools import partial

import torch.nn as nn

from legoml.nn.layers import (
    Conv1x1NormAct,
    DWConvNormAct,
    ScaledResidual,
)
from legoml.nn.primitives import ModuleCtor, identity
from legoml.nn.utils import make_divisible


class MBConv(nn.Sequential):
    def __init__(
        self,
        *,
        c_in: int,
        c_mid: int | None = None,
        c_out: int | None = None,
        s: int = 1,
        f_expand: int = 4,
        block1: ModuleCtor = Conv1x1NormAct,
        block2: ModuleCtor = DWConvNormAct,
        block3: ModuleCtor = partial(Conv1x1NormAct, act=nn.Identity),
        shortcut: ModuleCtor = nn.Identity,
        act: ModuleCtor = partial(nn.ReLU6, inplace=True),
        drop_path: float = 0.0,
    ):
        super().__init__()
        c_out = c_out or c_in
        c_mid = c_mid or make_divisible(c_in * f_expand)

        use_expand = c_mid != c_in
        apply_residual = s == 1 and c_in == c_out

        block1 = (
            block1(
                c_in=c_in,
                c_out=c_mid,
                s=1,
            )
            if use_expand
            else identity
        )
        block2 = block2(c_in=c_mid, c_out=c_mid, s=s)
        block3 = block3(c_in=c_mid, c_out=c_out, s=1)
        shortcut = shortcut(c_in=c_in, c_out=c_out, s=s)

        layers = [block1, block2, block3]

        if apply_residual:
            layers = [
                ScaledResidual(
                    fn=nn.Sequential(*layers),
                    shortcut=shortcut,
                    drop_prob=drop_path,
                )
            ]

        self.block = nn.Sequential(*layers, act())
