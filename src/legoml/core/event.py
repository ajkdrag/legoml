from enum import Enum


class Events(Enum):
    ENGINE_START = "engine_start"
    ENGINE_END = "engine_end"
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    STEP_START = "step_start"
    STEP_END = "step_end"
    BACKWARD_START = "backward_start"
    BACKWARD_END = "backward_end"


EVENT_TO_METHOD = {
    Events.ENGINE_START: "on_engine_start",
    Events.ENGINE_END: "on_engine_end",
    Events.EPOCH_START: "on_epoch_start",
    Events.EPOCH_END: "on_epoch_end",
    Events.STEP_START: "on_step_start",
    Events.STEP_END: "on_step_end",
    Events.BACKWARD_START: "on_backward_start",
    Events.BACKWARD_END: "on_backward_end",
}
