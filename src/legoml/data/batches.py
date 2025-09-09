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
        if device == Device.CUDA:
            fmt = torch.channels_last
            self.inputs = self.inputs.to(memory_format=fmt)
            self.targets = self.targets.to(memory_format=fmt)
        return self


@dataclass
class AutoencoderBatch:
    inputs: torch.Tensor  # (B, C, H, W)
