from typing import Any, Dict, Set

from legoml.core.context import Context
from legoml.core.state import EngineState


def implements(*method_names):
    def decorator(cls):
        cls._implemented_methods = set(method_names)
        return cls

    return decorator


class Callback:
    _implemented_methods: Set[str]

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

    def on_engine_start(self, context: Context, state: EngineState) -> None:
        pass

    def on_engine_end(self, context: Context, state: EngineState) -> None:
        pass

    def on_epoch_start(self, context: Context, state: EngineState) -> None:
        pass

    def on_epoch_end(self, context: Context, state: EngineState) -> None:
        pass

    def on_step_start(self, context: Context, state: EngineState, batch: Any) -> None:
        pass

    def on_step_end(self, context: Context, state: EngineState, batch: Any) -> None:
        pass

    def on_backward_start(self, context: Context, state: EngineState) -> None:
        pass

    def on_backward_end(self, context: Context, state: EngineState) -> None:
        pass
