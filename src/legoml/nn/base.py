import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass(kw_only=True)
class Node(ABC):
    """Base class with common functionality."""

    name: str | None = None

    def to_dict(self):
        return asdict(self)

    @abstractmethod
    def build(self, *args, **kwargs) -> nn.Module:
        pass


@dataclass
class NoopNode(Node):
    def build(self, *args, **kwargs) -> nn.Module:
        return nn.Identity()
