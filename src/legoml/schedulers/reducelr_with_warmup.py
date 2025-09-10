import math
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


class WarmupReduceLROnPlateau(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        max_epochs: int,
        warmup_epochs: int,
        max_lr: float,
        init_lr: float,
        scheduler: ReduceLROnPlateau,
        annealing: Literal["lin", "cos"] = "cos",
        last_epoch: int = 0,
    ) -> None:
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.last_epoch = last_epoch + 1
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.optimizer = optimizer
        self._scheduler = scheduler
        self.annealer = (
            self._annealing_linear if annealing == "lin" else self._annealing_cos
        )

        # Reset learning rates back to initial values
        for group in self.optimizer.param_groups:
            group["lr"] = init_lr

        self._last_lr = [init_lr for _ in optimizer.param_groups]

    @staticmethod
    def _annealing_cos(start, end, pct):
        """Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    @staticmethod
    def _annealing_linear(start, end, pct):
        """Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        return (end - start) * pct + start

    def step(self, metrics) -> None:  # type: ignore[override]
        print(f"Stepping at epoch {self.last_epoch}")
        current_epoch = self.last_epoch
        if current_epoch <= self.warmup_epochs:
            # Linear warmup phase
            scale = current_epoch / self.warmup_epochs
            for group in self.optimizer.param_groups:
                new_lr = self.annealer(
                    self.init_lr,
                    self.max_lr,
                    scale,
                )
                group["lr"] = new_lr
        elif current_epoch <= self.max_epochs:
            self._scheduler.step(metrics)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        self.last_epoch += 1
