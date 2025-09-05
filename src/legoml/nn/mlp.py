from functools import partial

import torch.nn as nn

from legoml.nn.types import ModuleCtor
from legoml.nn.utils import identity


class FCNormAct(nn.Sequential):
    """Linear layer with normalization and activation.

    Linear->BN->Act with Dropout support.

    Parameters
    ----------
    c_in : int
        Input features
    c_out : int, optional
        Output features. Defaults to c_in
    dropout : float, default=0.0
        Dropout probability after activation
    norm : Callable, optional
        Normalization. Defaults to BatchNorm2d
    act: Callable, optional
        Activation. Defaults to Relu
    """

    def __init__(
        self,
        *,
        c_in: int,
        c_out: int | None = None,
        norm: ModuleCtor = nn.BatchNorm1d,
        act: ModuleCtor = partial(nn.ReLU, inplace=True),
        dropout: float = 0.0,
    ):
        super().__init__()
        c_out = c_out or c_in

        # TODO: Should bias be set to False like ConvLayer?
        self.block = nn.Linear(c_in, c_out)
        self.norm = norm(c_out)
        self.act = act()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else identity
