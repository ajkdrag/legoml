from dataclasses import dataclass, asdict
from typing import Any, Dict, Literal
from abc import ABC, abstractmethod
from pathlib import Path
import torch
import torch.nn as nn
import json

from torch.utils.data import Dataset, DataLoader


@dataclass(kw_only=True)
class Node(ABC):
    """Base class with common functionality."""

    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path):
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


@dataclass
class OptimizerNode(Node):
    """Base class for optimizer."""

    @abstractmethod
    def build(self, params) -> torch.optim.Optimizer:
        pass


@dataclass
class SchedulerNode(Node):
    """Base class for scheduler."""

    @abstractmethod
    def build(self, optimizer: torch.optim.Optimizer) -> Any:
        pass


@dataclass
class DatasetForClassificationNode(Node):
    """Base class for supervised dataset for classification."""

    num_classes: int = 10

    @abstractmethod
    def build(self, split: Literal["train", "val", "test"]) -> Dataset:
        pass


@dataclass
class DataLoaderNode:
    """Base class for dataloader."""

    batch_size: int = 32
    pin_memory: bool = True
    num_workers: int = 4

    @abstractmethod
    def build(self, dataset: Dataset) -> DataLoader:
        pass


@dataclass
class ModelNode(Node):
    """Base class for models, layers and blocks."""

    @abstractmethod
    def build(self) -> nn.Module:
        pass
