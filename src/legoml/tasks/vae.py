from __future__ import annotations

from dataclasses import dataclass
from typing import List
import torch
import torch.nn.functional as F

from legoml.core.contracts.task import Task, Scalars
from legoml.core.contracts.metric import Metric
from legoml.data.batches import AutoencoderBatch


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # Mean over batch
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


@dataclass
class VAEReconstructionTask(Task):
    model: torch.nn.Module  # should return recon(flat), mu, logvar
    beta: float = 1.0
    recon_loss: str = "bce"  # or "mse"
    device: str = "cpu"
    metrics: List[Metric] | None = None

    def on_epoch_start(self, *, mode: str, epoch: int) -> None:
        if self.metrics:
            for m in self.metrics:
                m.reset()

    def on_epoch_end(self, *, mode: str, epoch: int) -> Scalars:
        if not self.metrics:
            return {}
        scalars: Scalars = {}
        for m in self.metrics:
            scalars.update(m.compute())
        return {f"{mode}/{k}": float(v) for k, v in scalars.items()}

    def _loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ):
        x_flat = x.view(x.size(0), -1)
        if self.recon_loss == "mse":
            rec = F.mse_loss(recon, x_flat, reduction="mean")
        else:
            rec = F.binary_cross_entropy(recon, x_flat, reduction="mean")
        kl = kl_divergence(mu, logvar)
        total = rec + self.beta * kl
        return total, {
            "loss/recon": float(rec.detach().item()),
            "loss/kl": float(kl.detach().item()),
            "loss/total": float(total.detach().item()),
        }

    def train_step(self, batch: AutoencoderBatch, *, step: int, epoch: int):
        x = batch.inputs.to(self.device)
        recon, mu, logvar = self.model(x)
        return self._loss(x, recon, mu, logvar)

    def eval_step(self, batch: AutoencoderBatch, *, step: int, epoch: int):
        x = batch.inputs.to(self.device)
        recon, mu, logvar = self.model(x)
        return self._loss(x, recon, mu, logvar)
