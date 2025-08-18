from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

Scalars = dict[str, float]


@dataclass
class Metric(ABC):
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(self, *, outputs: torch.Tensor, targets: torch.Tensor) -> None: ...

    @abstractmethod
    def compute(self) -> Scalars: ...
