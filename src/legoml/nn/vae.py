from dataclasses import dataclass, field
import torch
import torch.nn as nn
from .base import Node
from .mlp import MLPNode


@dataclass
class VAENode(Node):
    encoder: MLPNode = field(
        default_factory=lambda: MLPNode(
            dims=[256],
        )
    )
    decoder: MLPNode = field(
        default_factory=lambda: MLPNode(
            dims=[256, 28 * 28 * 1],
        )
    )
    mu: MLPNode = field(
        default_factory=lambda: MLPNode(
            dims=[16],
        )
    )
    logvar: MLPNode = field(
        default_factory=lambda: MLPNode(
            dims=[16],
        )
    )
    latent_dim: int = 16
    hidden: int = 256

    def build(self):
        return MLPVAE(self)


class MLPVAE(nn.Module):
    def __init__(self, config: VAENode):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim

        self.encoder = config.encoder.build()
        self.decoder = config.decoder.build()
        self.mu = config.mu.build()
        self.logvar = config.logvar.build()

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        x = self.decoder(z)
        return torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon = recon.view(x.size(0), -1)
        return recon, mu, logvar
