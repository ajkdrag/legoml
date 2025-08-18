from dataclasses import dataclass, field
import torch.nn as nn
from .base import Node
from .activations import ReluNode
from .pooling import MaxPool2dNode


@dataclass
class ConvBlockNode(Node):
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    batch_norm: bool = True
    activation: Node = field(default_factory=ReluNode)
    dropout: float = 0.0
    pooling: Node = field(default_factory=MaxPool2dNode)

    def build(self) -> nn.Module:
        return ConvBlock(self)


class ConvBlock(nn.Sequential):
    def __init__(self, config: ConvBlockNode):
        self.config = config

        layers = []
        layers.append(
            nn.Conv2d(
                config.in_channels,
                config.out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding,
            )
        )
        if config.batch_norm:
            layers.append(nn.BatchNorm2d(config.out_channels))

        layers.append(config.activation.build())
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))

        layers.append(config.pooling.build())
        super().__init__(*layers)
