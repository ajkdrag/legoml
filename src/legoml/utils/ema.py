import copy
from typing import cast

import torch
import torch.nn as nn


class ModelEma:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self._model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self._model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self._model.state_dict().items():
            v = cast(torch.Tensor, v)
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k], alpha=1 - d)
