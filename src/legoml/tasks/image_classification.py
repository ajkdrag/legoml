from dataclasses import dataclass
import torch

from legoml.core.constants import Mode, Device
from legoml.core.contracts.task import Task, Scalars
from legoml.core.contracts.objective import Objective
from legoml.core.contracts.metric import Metric
from legoml.data.batches import ClassificationBatch


@dataclass
class ImageClassificationTask(Task):
    model: torch.nn.Module
    objective: Objective
    metrics: list[Metric]
    device: Device = Device.CPU

    def on_epoch_start(self, *, mode: Mode, epoch: int) -> None:
        for m in self.metrics:
            m.reset()

    def on_epoch_end(self, *, mode: Mode, epoch: int) -> Scalars:
        logs: Scalars = {}
        for m in self.metrics:
            logs.update(m.compute())
        return {f"{mode.value}/{k}": float(v) for k, v in logs.items()}

    def _process(
        self, batch: ClassificationBatch, *, mode: Mode
    ) -> tuple[torch.Tensor, Scalars]:
        batch = batch.to(self.device)
        loss, scalars = self.objective(self.model, batch, mode=mode)
        outputs = self.model(batch.inputs)
        for m in self.metrics:
            m.update(
                outputs=outputs.detach().cpu(),
                targets=batch.targets.detach().cpu(),
            )

        return loss, scalars

    def train_step(
        self, batch: ClassificationBatch, *, step: int, epoch: int
    ) -> tuple[torch.Tensor, Scalars]:
        return self._process(batch, mode=Mode.TRAIN)

    def eval_step(
        self, batch: ClassificationBatch, *, step: int, epoch: int
    ) -> tuple[torch.Tensor, Scalars]:
        return self._process(batch, mode=Mode.EVAL)
