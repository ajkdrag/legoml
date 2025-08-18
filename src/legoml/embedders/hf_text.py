from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import torch


@dataclass
class HFTextEmbedder:
    model_name: str = "prajjwal1/bert-tiny"
    device: str = "cpu"
    pooling: str = "cls"  # or "mean"

    def __post_init__(self):
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
        except Exception as e:
            raise ImportError(
                "Please `pip install transformers` to use HFTextEmbedder"
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            list(texts), padding=True, truncation=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model(**inputs)
        hidden = out.last_hidden_state  # (B, T, H)
        if self.pooling == "cls":
            return hidden[:, 0, :].detach().cpu()
        elif self.pooling == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            return (summed / counts).detach().cpu()
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
