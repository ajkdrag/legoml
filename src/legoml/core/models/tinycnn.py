import torchsummary
from dataclasses import dataclass, field
from typing import List
import torch

import torch.nn as nn

from legoml.core.base import ModelNode
from legoml.core.blocks.convs import ConvBlockNode
from legoml.core.blocks.activations import ReluNode


@dataclass
class TinyCNNNode(ModelNode):
    input_channels: int = 1
    num_classes: int = 1

    conv_blocks: List[ConvBlockNode] = field(
        default_factory=lambda: [
            ConvBlockNode(in_channels=1, out_channels=32, kernel_size=3),
            ConvBlockNode(in_channels=32, out_channels=64, kernel_size=3),
            ConvBlockNode(in_channels=64, out_channels=128, kernel_size=3),
        ]
    )

    fc_features: List[int] = field(default_factory=lambda: [256, 128])
    fc_activation: ModelNode = field(default_factory=ReluNode)
    fc_dropout: float = 0.5

    def build(self) -> nn.Module:
        return TinyCNN(self)


class TinyCNN(nn.Module):
    def __init__(self, config: TinyCNNNode):
        super().__init__()
        self.config = config

        conv_layers = [block.build() for block in config.conv_blocks]
        self.features = nn.Sequential(*conv_layers)

        fc_layers = []
        for idx, fc_size in enumerate(config.fc_features):
            fc_layers.append(nn.LazyLinear(fc_size))
            fc_layers.append(config.fc_activation.build())
            if config.fc_dropout > 0:
                fc_layers.append(nn.Dropout(config.fc_dropout))

        fc_layers.append(nn.LazyLinear(config.num_classes))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
