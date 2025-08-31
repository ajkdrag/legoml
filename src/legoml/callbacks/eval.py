from torch.utils.data import DataLoader
from legoml.core.callback import Callback, implements
from legoml.core.engine import Engine
from legoml.core.context import Context
from legoml.core.state import EngineState
from legoml.utils.log import get_logger


logger = get_logger(__name__)


@implements("on_epoch_end")
class EvalOnEpochEndCallback(Callback):
    def __init__(
        self,
        evaluator: Engine,
        dataloader: DataLoader,
        frequency: int = 1,
    ):
        self.evaluator = evaluator
        self.dataloader = dataloader
        self.frequency = frequency

    def on_epoch_end(self, context: Context, state: EngineState) -> None:
        if state.epoch % self.frequency != 0:
            return

        self.evaluator.loop(dataloader=self.dataloader, max_epochs=1)
        logger.info(
            "Evaluation complete",
            step=state.epoch,
        )
