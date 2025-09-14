from typing import Any, Callable

from torch.utils.data import DataLoader

from legoml.core.callback import Callback
from legoml.core.context import Context
from legoml.core.event import EVENT_TO_METHOD, Events
from legoml.core.state import EngineState
from legoml.core.step_output import StepOutput
from legoml.utils.io import load_checkpoint


class Engine:
    def __init__(
        self,
        fn: Callable[["Engine", Any, Context], StepOutput],
        context: Context,
        state: EngineState | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.fn = fn
        self.context = context
        self.state = state or EngineState()
        self.callbacks = callbacks or []

    def fire(self, event: Events, **kwargs) -> None:
        method = EVENT_TO_METHOD[event]
        for cb in self.callbacks:
            fn = getattr(cb, method, None)
            if callable(fn):
                fn(
                    context=self.context,
                    state=self.state,
                    **kwargs,
                )

    def add_event_handler(self, event: Events, handler: Callable[..., None] | Callback):
        if not isinstance(handler, Callback):
            callback = Callback()
            setattr(callback, EVENT_TO_METHOD[event], handler)
        else:
            callback = handler
        self.callbacks.append(callback)

    def loop(self, dataloader: DataLoader | None = None, max_epochs: int | None = None):
        self.state.max_epochs = max_epochs or self.state.max_epochs
        self.state.dataloader = dataloader or self.state.dataloader
        if self.state.dataloader is None:
            raise ValueError("No dataloader passed or found in state")

        self.fire(Events.ENGINE_START)

        for epoch in range(self.state.epoch, self.state.max_epochs + 1):
            self.state.epoch = epoch
            self.fire(Events.EPOCH_START)

            for idx, batch in enumerate(self.state.dataloader, start=1):
                self.state.local_step = idx
                self.fire(Events.STEP_START, batch=batch)
                op = self.fn(self, batch, self.context)

                self.state.output = op
                self.fire(Events.STEP_END, batch=batch)
                self.state.local_step += 1
                self.state.global_step += 1
                if self.state.should_stop:
                    break

            self.fire(Events.EPOCH_END)
            if self.state.should_stop:
                break

        self.fire(Events.ENGINE_END)

    def load_checkpoint(self, checkpoint_path: str):
        ckpt = load_checkpoint(checkpoint_path, device=self.context.device)
        model_sd = ckpt.get("model")
        if model_sd is not None:
            self.context.model.load_state_dict(model_sd)
        opt_sd = ckpt.get("optimizer")
        if opt_sd is not None and self.context.optimizer is not None:
            self.context.optimizer.load_state_dict(opt_sd)  # type: ignore[arg-type]
        sch_sd = ckpt.get("scheduler")
        if sch_sd is not None and self.context.scheduler is not None:
            self.context.scheduler.load_state_dict(sch_sd)  # type: ignore[arg-type]
        scaler_sd = ckpt.get("scaler")
        if scaler_sd is not None and self.context.scaler is not None:
            self.context.scaler.load_state_dict(scaler_sd)  # type: ignore[arg-type]

        if self.state is not None:
            self.state.epoch = int(ckpt.get("epoch", self.state.epoch))
            self.state.global_step = int(
                ckpt.get("global_step", self.state.global_step)
            )
            self.state.local_step = int(ckpt.get("local_step", self.state.local_step))
            self.state.max_epochs = int(ckpt.get("max_epochs", self.state.max_epochs))

        return ckpt

    def to_dict(self):
        return {
            "context": self.context.to_dict(),
            "state": self.state.to_dict(),
            "callbacks": [cb.__class__.__name__ for cb in self.callbacks],
        }
