from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader

from legoml.core.constants import Mode
from legoml.core.contracts.task import Task
from legoml.core.contracts.callback import Callback
from legoml.core.strategies.single_device import SingleDeviceStrategy
from legoml.utils.logging import get_logger
from legoml.core.state import FitState


logger = get_logger("legoml.trainer")


@dataclass
class Trainer:
    task: Task
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    strategy: SingleDeviceStrategy
    callbacks: list[Callback]
    max_epochs: int = 1
    log_every_n_steps: int = 50

    def eval(self, state: FitState, eval_loader: DataLoader):
        self.task.model.eval()
        self.task.on_epoch_start(mode=Mode.EVAL, epoch=state.epoch)
        for cb in self.callbacks:
            cb.on_eval_epoch_start(state)

        with torch.no_grad():
            for step, batch in enumerate(eval_loader, start=1):
                state.step = step
                for cb in self.callbacks:
                    cb.on_eval_batch_start(state)

                loss, logs = self.task.eval_step(
                    batch,
                    step=step,
                    epoch=state.epoch,
                )
                state.last_eval_loss = float(loss.detach().item())
                state.last_eval = {**logs}
                for cb in self.callbacks:
                    cb.on_eval_batch_end(state)

        state.eval_epoch = self.task.on_epoch_end(
            mode=Mode.EVAL,
            epoch=state.epoch,
        )
        for cb in self.callbacks:
            cb.on_eval_epoch_end(state)

    def fit(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader | None = None,
    ):
        # move to device, amp etc
        self.task.model = self.strategy.prepare(self.task.model)

        state = FitState(
            epoch=0,
            step=0,
            global_step=0,
            model=self.task.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            task=self.task,
            device=self.strategy.device,
            train_total=len(train_loader),
            eval_total=len(eval_loader) if eval_loader else None,
        )

        for cb in self.callbacks:
            cb.on_fit_start(state)

        for epoch in range(1, self.max_epochs + 1):
            state.epoch = epoch

            self.task.model.train()
            self.task.on_epoch_start(mode=Mode.TRAIN, epoch=epoch)
            for cb in self.callbacks:
                cb.on_train_epoch_start(state)

            for step, batch in enumerate(train_loader, start=1):
                state.global_step += 1
                state.step = step
                for cb in self.callbacks:
                    cb.on_train_batch_start(state)

                with self.strategy.autocast():
                    loss, logs = self.task.train_step(
                        batch,
                        step=step,
                        epoch=epoch,
                    )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                if step % self.log_every_n_steps == 0:
                    logger.info(
                        "train_step",
                        epoch=epoch,
                        step=step,
                        loss=float(loss.detach().item()),
                        **{f"train/{k}": v for k, v in logs.items()},
                    )

                state.last_train_loss = float(loss.detach().item())
                state.last_train = {**logs}
                for cb in self.callbacks:
                    cb.on_train_batch_end(state)

            state.train_epoch = self.task.on_epoch_end(
                mode=Mode.TRAIN,
                epoch=epoch,
            )
            for cb in self.callbacks:
                cb.on_train_epoch_end(state)

            if eval_loader is not None:
                self.eval(state, eval_loader)

            if self.scheduler is not None:
                self.scheduler.step()

        for cb in self.callbacks:
            cb.on_fit_end(state)
        return state
