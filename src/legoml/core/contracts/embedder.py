from typing import Protocol, runtime_checkable, Sequence
import torch
from PIL import Image


@runtime_checkable
class TextEmbedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...  # (B, dim)


@runtime_checkable
class ImageEmbedder(Protocol):
    dim: int

    def encode(self, images: Sequence[Image.Image]) -> torch.Tensor: ...  # (B, dim)
