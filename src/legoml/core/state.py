from dataclasses import dataclass, field
from typing import Iterable
from legoml.core.step_output import StepOutput


@dataclass(kw_only=True)
class EngineState:
    epoch: int = 0
    max_epochs: int = 1
    global_step: int = 0  # global step
    local_step: int = 0  # local step
    output: StepOutput = field(default_factory=StepOutput)  # output of the last step
    metrics: dict[str, float] = field(default_factory=dict)
    dataloader: Iterable | None = None
    should_stop: bool = False
