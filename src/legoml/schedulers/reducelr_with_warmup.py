import math
from typing import Literal, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


class WarmupThenPlateau(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        max_epochs: int,
        warmup_epochs: int,
        max_lr: float,
        init_lr: float,
        plateau_kwargs: Optional[
            dict
        ] = None,  # e.g. {"mode": "min", "patience": 10, "factor": 0.1}
        annealing: Literal["lin", "cos"] = "cos",
        last_epoch: int = -1,
    ) -> None:
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.annealer = (
            self._annealing_linear if annealing == "lin" else self._annealing_cos
        )

        for g in optimizer.param_groups:
            g["lr"] = self.init_lr

        self._plateau_kwargs = plateau_kwargs or {}
        self._plateau: Optional[ReduceLROnPlateau] = None
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _annealing_cos(start, end, pct):
        return end + (start - end) * (0.5 * (1 + math.cos(math.pi * pct)))

    @staticmethod
    def _annealing_linear(start, end, pct):
        return (end - start) * pct + start

    def step(self, metrics: float | None = None):  # type: ignore[override]
        epoch = self.last_epoch + 1

        if epoch <= self.warmup_epochs:
            pct = epoch / max(1, self.warmup_epochs)
            lr = self.annealer(self.init_lr, self.max_lr, pct)
            for g in self.optimizer.param_groups:
                g["lr"] = lr
        else:
            if self._plateau is None:
                # Create at handover so patience starts counting *after* warmup.
                self._plateau = ReduceLROnPlateau(
                    self.optimizer, **self._plateau_kwargs
                )
            if not metrics:
                raise ValueError("Metrics not provided to step call")
            self._plateau.step(metrics)

        self._last_lr = [g["lr"] for g in self.optimizer.param_groups]
        self.last_epoch = epoch
