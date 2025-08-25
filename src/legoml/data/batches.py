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
