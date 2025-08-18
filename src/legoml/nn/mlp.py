from dataclasses import dataclass, field
import torch.nn as nn
from .base import Node
from .activations import ReluNode


@dataclass
class MLPNode(Node):
    dims: list[int] = field(default_factory=list)
    activation: Node = field(default_factory=ReluNode)
    last_activation: Node = field(default_factory=ReluNode)

    def build(self):
        return MLP(config=self)


class MLP(nn.Sequential):
    # using LazyLinear to avoid having to specify input size
    def __init__(self, config: MLPNode):
        self.config = config
        layers = []
        for idx, dim in enumerate(config.dims):
            layers.append(nn.LazyLinear(dim))
            if idx == len(config.dims) - 1:
                layers.append(config.last_activation.build())
            else:
                layers.append(config.activation.build())

        super().__init__(*layers)
