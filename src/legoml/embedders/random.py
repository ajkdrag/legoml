from dataclasses import dataclass
from typing import Sequence
import torch
from PIL import Image


@dataclass
class RandomTextEmbedder:
    dim: int = 64
    seed: int = 42

    def __post_init__(self):
        self._gen = torch.Generator().manual_seed(self.seed)

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.randn(len(texts), self.dim, generator=self._gen)


@dataclass
class RandomImageEmbedder:
    dim: int = 64
    seed: int = 123

    def __post_init__(self):
        self._gen = torch.Generator().manual_seed(self.seed)

    def encode(self, images: Sequence[Image.Image]) -> torch.Tensor:
        return torch.randn(len(images), self.dim, generator=self._gen)
