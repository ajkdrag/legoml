import copy
from typing import cast

import torch
import torch.nn as nn


class ModelEma:
    _model: nn.Module

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
    ):
        self.decay = decay
        self.num_updates = 0
        self.copy(model)

    def copy(self, other: nn.Module):
        self._model = copy.deepcopy(other).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, *, model: nn.Module):
        self.num_updates += 1
        d = self.decay
        msd = model.state_dict()

        for k, v in self._model.state_dict().items():
            v = cast(torch.Tensor, v)
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k], alpha=1 - d)

    @torch.no_grad()
    def copy_bn_from(self, model: nn.Module):
        """Copy BN buffers from the online model (useful early in training)."""
        for (_, m1), (_, m2) in zip(self._model.named_modules(), model.named_modules()):
            if isinstance(m1, nn.modules.batchnorm._BatchNorm) and isinstance(
                m2, nn.modules.batchnorm._BatchNorm
            ):
                if m1.running_mean is not None and m2.running_mean is not None:
                    m1.running_mean.copy_(m2.running_mean)
                if m1.running_var is not None and m2.running_var is not None:
                    m1.running_var.copy_(m2.running_var)
                if (
                    m1.num_batches_tracked is not None
                    and m2.num_batches_tracked is not None
                ):
                    m1.num_batches_tracked.copy_(m2.num_batches_tracked)
