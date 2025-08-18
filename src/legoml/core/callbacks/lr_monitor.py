from dataclasses import dataclass
from typing_extensions import override
from legoml.core.contracts.callback import CallbackBase
from legoml.utils.logging import get_logger
from legoml.core.state import FitState


logger = get_logger("legoml.callbacks.lr")


@dataclass
class LearningRateMonitor(CallbackBase):
    @override
    def on_train_epoch_end(self, state: FitState):
        opt = state.optimizer
        if not opt:
            return
        try:
            lrs = [pg.get("lr", None) for pg in opt.param_groups]
            logger.info("lr", values=lrs)
        except Exception:
            pass
