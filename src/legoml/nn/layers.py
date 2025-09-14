import math
from functools import partial

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from legoml.nn.primitives import (
    ModuleCtor,
    identity,
)
from legoml.nn.utils import autopad


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
