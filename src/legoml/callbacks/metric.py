from legoml.core.context import Context
from legoml.core.metric import Metric
from legoml.core.callback import Callback, implements
from legoml.core.state import EngineState
from legoml.utils.log import get_logger

logger = get_logger(__name__)


@implements("on_epoch_start", "on_step_end", "on_epoch_end")
class MetricsCallback(Callback):
    def __init__(self, metrics: list[Metric]):
        self.metrics = metrics

    def on_epoch_start(self, context: Context, state: EngineState):
        for metric in self.metrics:
            metric.reset()

    def on_step_end(
        self,
        context: Context,
        state: EngineState,
        batch,
    ):
        for metric in self.metrics:
            try:
                metric.update(output=state.output)
            except Exception as e:
                logger.error(
                    f"Failed to update metric {metric.__class__.__name__}: {e}"
                )

    def on_epoch_end(self, context: Context, state: EngineState):
        state.metrics = {}
        for metric in self.metrics:
            try:
                result = metric.compute()
                state.metrics.update(result)
                logger.info(f"Metric: {result}")
            except Exception as e:
                logger.error(f"Failed to compute metric {metric.name}: {e}")
