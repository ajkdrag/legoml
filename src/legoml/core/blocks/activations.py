from dataclasses import dataclass
from legoml.core.base import ModelNode
import torch


@dataclass
class ReluNode(ModelNode):
    """Rectified Linear Unit (ReLU) activation function."""

    def build(self) -> torch.nn.Module:
        return torch.nn.ReLU(inplace=True)


@dataclass
class SigmoidNode(ModelNode):
    """Sigmoid activation function."""

    def build(self) -> torch.nn.Module:
        return torch.nn.Sigmoid(inplace=True)


@dataclass
class TanhNode(ModelNode):
    """Hyperbolic Tangent (Tanh) activation function."""

    def build(self) -> torch.nn.Module:
        return torch.nn.Tanh(inplace=True)
