from dataclasses import dataclass
import torch
from legoml.core.constants import Device


@dataclass
class ClassificationBatch:
    inputs: torch.Tensor  # (B, C, H, W)
    targets: torch.Tensor  # (B,)

    def to(self, device: Device):
        self.inputs = self.inputs.to(device.value)
        self.targets = self.targets.to(device.value)
        return self


@dataclass
class AutoencoderBatch:
    inputs: torch.Tensor  # (B, C, H, W)


def classification_collate(batch: list[tuple[torch.Tensor, int]]):
    inputs = torch.stack([x for x, _ in batch], dim=0)
    targets = torch.tensor([y for _, y in batch], dtype=torch.int64)
    return ClassificationBatch(inputs=inputs, targets=targets)


def autoencoder_collate(batch: list[tuple[torch.Tensor, int]]):
    inputs = torch.stack([x for x, _ in batch], dim=0)
    return AutoencoderBatch(inputs=inputs)
