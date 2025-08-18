from __future__ import annotations

from typing import Iterable
import torch

from legoml.core.contracts.objective import Objective, Scalars
from legoml.core.constants import Mode


class WeightedSum(Objective):
    def __init__(self, items: Iterable[tuple[Objective, float]]):
        self.items = list(items)

    def __call__(
        self, model: torch.nn.Module, batch, *, mode: Mode
    ) -> tuple[torch.Tensor, Scalars]:
        total = None
        logs: Scalars = {}
        for obj, weight in self.items:
            loss, obj_logs = obj(model, batch, mode=mode)
            total = loss * weight if total is None else total + loss * weight
            for k, v in obj_logs.items():
                logs[k] = logs.get(k, 0.0) + float(v) * weight
        if total is None:
            raise ValueError("WeightedSum received no objectives")
        logs["loss/total"] = float(total.detach().item())
        return total, logs
