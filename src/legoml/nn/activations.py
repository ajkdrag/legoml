from dataclasses import dataclass
from .base import Node
import torch


@dataclass
class ReluNode(Node):
    """Rectified Linear Unit (ReLU) activation function."""

    inplace: bool = True

    def build(self) -> torch.nn.Module:
        return torch.nn.ReLU(inplace=self.inplace)


@dataclass
class SigmoidNode(Node):
    """Sigmoid activation function."""

    inplace: bool = True

    def build(self) -> torch.nn.Module:
        return torch.nn.Sigmoid(inplace=self.inplace)


@dataclass
class TanhNode(Node):
    """Hyperbolic Tangent (Tanh) activation function."""

    inplace: bool = True

    def build(self) -> torch.nn.Module:
        return torch.nn.Tanh(inplace=self.inplace)
