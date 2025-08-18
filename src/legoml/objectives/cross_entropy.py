from typing import Any

import torch
import torch.nn.functional as F

from legoml.core.contracts.objective import Objective, Scalars
from legoml.core.constants import Mode


class CrossEntropyObjective(Objective):
    def __call__(
        self,
        model: torch.nn.Module,
        batch: Any,
        *,
        mode: Mode,
    ) -> tuple[torch.Tensor, Scalars]:
        # Expect a dataclass with 'inputs' and 'targets' attributes
        inputs = getattr(batch, "inputs")
        targets = getattr(batch, "targets")
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        return loss, {"loss/xent": float(loss.detach().item())}
