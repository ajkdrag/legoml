from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from legoml.core.step_output import StepOutput


@dataclass(kw_only=True)
class EngineState:
    epoch: int = 0
    max_epochs: int = 1
    global_step: int = 0  # global step
    local_step: int = 0  # local step
    output: StepOutput = field(default_factory=StepOutput)  # output of the last step
    metrics: dict[str, float] = field(default_factory=dict)
    dataloader: DataLoader | None = None
    should_stop: bool = False

    def reset(self):
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.output = StepOutput()
        self.metrics = {}
        self.dataloader = None
        self.should_stop = False

    def to_dict(self):
        result = {
            "epoch": self.epoch,
            "max_epochs": self.max_epochs,
            "global_step": self.global_step,
            "local_step": self.local_step,
            "metrics": self.metrics,
            "should_stop": self.should_stop,
        }

        if self.dataloader is not None:
            result["dataset"] = self.dataloader.dataset.__class__.__name__
            result["batch_size"] = self.dataloader.batch_size
            result["num_batches"] = len(self.dataloader)

        return result
