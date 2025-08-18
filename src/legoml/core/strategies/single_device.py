from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from legoml.core.constants import Device
import torch


@dataclass
class SingleDeviceStrategy:
    device: Device = Device.CPU
    use_amp: bool = False

    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        return model.to(self.device.value)

    def autocast(self):
        if self.use_amp and self.device == Device.CUDA:
            return torch.autocast(device_type=self.device.value)
        return nullcontext()
