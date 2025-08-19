from typing import Callable, Any

from legoml.core.callback import Callback
from legoml.core.context import Context
from legoml.core.event import EVENT_TO_METHOD, Events
from legoml.core.state import EngineState
from legoml.core.step_output import StepOutput


class Engine:
    def __init__(
        self,
        fn: Callable[["Engine", Any, Context], StepOutput],
        context: Context,
        callbacks: list[Callback] | None = None,
    ) -> None:
        """
        Args:
            fn: Step function that processes a batch. Signature: fn(engine, batch, context) -> StepOutput
            context: Shared training context containing model, optimizer, etc.
            callbacks: List of callbacks to attach to the engine
        """
        self.fn = fn
        self.context = context
        self.state = EngineState()
        self.callbacks = callbacks or []

    def fire(self, event: Events, **kwargs) -> None:
        method = EVENT_TO_METHOD[event]
        for cb in self.callbacks:
            if method in cb._implemented_methods:
                getattr(cb, method)(
                    context=self.context,
                    state=self.state,
                    **kwargs,
                )

    def loop(self, dataloader, max_epochs: int):
        self.state.max_epochs = max_epochs
        self.state.dataloader = dataloader
        self.fire(Events.ENGINE_START)

        for epoch in range(self.state.epoch, max_epochs):
            self.state.epoch = epoch
            self.fire(Events.EPOCH_START)

            for idx, batch in enumerate(dataloader):
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
