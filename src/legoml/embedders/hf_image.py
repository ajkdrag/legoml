from __future__ import annotations

from dataclasses import dataclass
from legoml.core.constants import Device
from typing import Sequence
from PIL import Image
import torch


@dataclass
class HFImageEmbedder:
    model_name: str = "hf-internal-testing/tiny-random-vit"
    device: Device = Device.CPU
    pooling: str = "cls"  # depends on model

    def __post_init__(self):
        try:
            from transformers import AutoImageProcessor, AutoModel  # type: ignore
        except Exception as e:
            raise ImportError(
                "Please `pip install transformers` to use HFImageEmbedder"
            ) from e

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device.value)
        hidden = getattr(self.model.config, "hidden_size", None) or getattr(
            self.model.config, "projection_dim", None
        )
        if hidden is None:
            raise ValueError(
                "Could not infer image embedding dimension from model config"
            )
        self.dim = int(hidden)

    @torch.no_grad()
    def encode(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(list(images), return_tensors="pt").to(self.model.device)
        out = self.model(**inputs)
        # Attempt to find a reasonable pooled representation
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            x = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
            x = hidden[:, 0, :] if self.pooling == "cls" else hidden.mean(dim=1)
        elif hasattr(out, "image_embeds"):
            x = out.image_embeds
        else:
            raise ValueError("Unsupported output structure for image model")
        return x.detach().cpu()
