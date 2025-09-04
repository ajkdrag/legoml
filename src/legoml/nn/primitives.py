from functools import partial
from typing import Callable

import torch.nn as nn

ModuleCtor = Callable[..., nn.Module]

Relu: ModuleCtor = partial(nn.ReLU, inplace=True)
identity = nn.Identity()
