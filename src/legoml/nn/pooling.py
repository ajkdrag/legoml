from dataclasses import dataclass
from .base import Node
import torch


@dataclass
class MaxPool2dNode(Node):
    kernel_size: int = 2
    stride: int = 2
    padding: int = 0
    dilation: int = 1

    def build(self) -> torch.nn.Module:
        return torch.nn.MaxPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


@dataclass
class AvgPool2dNode(Node):
    kernel_size: int = 2
    stride: int = 2
    padding: int = 0

    def build(self) -> torch.nn.Module:
        return torch.nn.AvgPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
