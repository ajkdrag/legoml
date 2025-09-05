from typing import Callable

import torch.nn as nn

ModuleCtor = Callable[..., nn.Module]
