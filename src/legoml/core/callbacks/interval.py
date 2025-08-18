from dataclasses import dataclass
from typing import Callable, Any
from typing_extensions import override
from legoml.core.contracts.callback import CallbackBase
from legoml.core.state import FitState


@dataclass
class EpochIntervalHook(CallbackBase):
    every_n_epochs: int
    fn: Callable[[FitState], None]
    when: str = "on_train_epoch_end"  # or on_eval_epoch_end

    @override
    def on_train_epoch_end(self, state: FitState):
        if self.when != "on_train_epoch_end":
            return
        epoch = state.epoch
        if epoch % self.every_n_epochs == 0:
            self.fn(state)

    @override
    def on_eval_epoch_end(self, state: FitState):
        if self.when != "on_eval_epoch_end":
            return
        epoch = state.epoch
        if epoch % self.every_n_epochs == 0:
            self.fn(state)


@dataclass
class LinearAttrWarmup(CallbackBase):
    target: Any
    attr: str
    start: float
    end: float
    warmup_epochs: int

    @override
    def on_fit_start(self, state: FitState):
        setattr(self.target, self.attr, self.start)

    @override
    def on_train_epoch_end(self, state: FitState):
        e = state.epoch
        if e >= self.warmup_epochs:
            value = self.end
        else:
            value = self.start + (self.end - self.start) * (
                e / max(1, self.warmup_epochs)
            )
        setattr(self.target, self.attr, value)
