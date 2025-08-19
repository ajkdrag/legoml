from dataclasses import dataclass, field
from typing import List

import torch.nn as nn

from legoml.nn.activations import ReluNode
from legoml.nn.base import Node, NoopNode
from legoml.nn.convs import ConvBlockNode
from legoml.nn.mlp import MLPNode


@dataclass
class TinyCNNNode(Node):
    input_channels: int = 1
    conv_blocks: List[ConvBlockNode] = field(
        default_factory=lambda: [
            ConvBlockNode(in_channels=1, out_channels=32, kernel_size=3),
            ConvBlockNode(in_channels=32, out_channels=64, kernel_size=3),
            ConvBlockNode(in_channels=64, out_channels=128, kernel_size=3),
        ]
    )

    mlp: MLPNode = field(
        default_factory=lambda: MLPNode(
            dims=[128, 1],
            activation=ReluNode(),
            last_activation=NoopNode(),
        )
    )

    def build(self):
        return TinyCNN(self)


class TinyCNN(nn.Sequential):
    def __init__(self, config: TinyCNNNode):
        self.config = config
        layers = []
        for block in config.conv_blocks:
            layers.append(block.build())
        layers.append(nn.Flatten())
        layers.append(config.mlp.build())
        super().__init__(*layers)
