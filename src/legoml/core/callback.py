from typing import Any, Dict, Set

from legoml.core.context import Context
from legoml.core.state import EngineState


def implements(*method_names):
    """Decorator to mark which callback methods a class implements."""

    def decorator(cls):
        cls._implemented_methods = set(method_names)
        return cls

    return decorator


class Callback:
    """
    Protocol defining the callback interface for engine events.
    All methods are optional - callbacks only need to implement the events they care about.
    """

    _implemented_methods: Set[str]

    def state_dict(self) -> Dict[str, Any]:
        """Return callback state for checkpointing."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        pass

    def on_engine_start(self, context: Context, state: EngineState) -> None:
        """Called when the engine starts training."""
        pass

    def on_engine_end(self, context: Context, state: EngineState) -> None:
        """Called when the engine finishes training."""
        pass

    def on_epoch_start(self, context: Context, state: EngineState) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, context: Context, state: EngineState) -> None:
        """Called at the end of each epoch."""
        pass

    def on_step_start(self, context: Context, state: EngineState, batch: Any) -> None:
        """Called before processing each batch."""
        pass

    def on_step_end(self, context: Context, state: EngineState, batch: Any) -> None:
        """Called after processing each batch. Step outputs available in state.outputs."""
        pass

    def on_backward_start(self, context: Context, state: EngineState) -> None:
        """Called before backward pass."""
        pass

    def on_backward_end(self, context: Context, state: EngineState) -> None:
        """Called after backward pass."""
        pass
