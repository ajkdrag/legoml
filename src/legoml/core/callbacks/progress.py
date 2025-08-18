from dataclasses import dataclass
from typing_extensions import override
from tqdm import tqdm
from legoml.core.contracts.callback import CallbackBase
from legoml.core.state import FitState


@dataclass
class TQDMProgressBar(CallbackBase):
    leave: bool = False

    def __post_init__(self):
        self._train_bar = None
        self._val_bar = None

    @override
    def on_train_epoch_start(self, state: FitState):
        total = state.train_total
        self._train_bar = tqdm(total=total, desc="train", leave=self.leave)

    @override
    def on_train_batch_end(self, state: FitState):
        if self._train_bar is not None:
            self._train_bar.update(1)
            if state.last_train_loss is not None:
                self._train_bar.set_postfix({"loss": f"{state.last_train_loss:.4f}"})

    @override
    def on_train_epoch_end(self, state: FitState):
        if self._train_bar is not None:
            self._train_bar.close()
            self._train_bar = None

    @override
    def on_eval_epoch_start(self, state: FitState):
        total = state.eval_total
        self._val_bar = tqdm(total=total, desc="val", leave=self.leave)

    @override
    def on_eval_batch_end(self, state: FitState):
        if self._val_bar is not None:
            self._val_bar.update(1)
            if state.last_eval_loss is not None:
                self._val_bar.set_postfix({"loss": f"{state.last_eval_loss:.4f}"})

    @override
    def on_eval_epoch_end(self, state: FitState):
        if self._val_bar is not None:
            self._val_bar.close()
            self._val_bar = None
