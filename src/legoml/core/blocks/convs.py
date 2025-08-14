from dataclasses import dataclass, field
import torch.nn as nn
from legoml.core.base import ModelNode
from legoml.core.blocks.activations import ReluNode
from legoml.core.blocks.pooling import MaxPool2dNode


@dataclass
class ConvBlockNode(ModelNode):
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    batch_norm: bool = True
    activation: ModelNode | None = field(default_factory=ReluNode)
    dropout: float = 0.0
    pooling: ModelNode | None = field(default_factory=MaxPool2dNode)

    def build(self) -> nn.Module:
        layers = []
        layers.append(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        )
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(self.out_channels))

        if self.activation:
            layers.append(self.activation.build())

        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))

        if self.pooling:
            layers.append(self.pooling.build())

        return nn.Sequential(*layers)
