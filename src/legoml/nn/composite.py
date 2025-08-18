import torch
import torch.nn as nn


class EncoderForClassification(nn.Module):
    """Compose an encoder with a classification head."""

    def __init__(
        self, encoder: nn.Module, head: nn.Module, *, freeze_encoder: bool = True
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)
