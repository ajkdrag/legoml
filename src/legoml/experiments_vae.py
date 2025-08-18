from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from legoml.core.loops.trainer import Trainer
from legoml.core.strategies.single_device import SingleDeviceStrategy
from legoml.core.callbacks.progress import TQDMProgressBar
from legoml.core.callbacks.checkpoint import ModelCheckpoint
from legoml.core.callbacks.lr_monitor import LearningRateMonitor
from legoml.core.callbacks.interval import LinearAttrWarmup

from legoml.nn.vae import MLPVAE, VAEConfig
from legoml.tasks.vae import VAEReconstructionTask
from legoml.data.mnist import MNISTConfig, build_mnist
from legoml.data.batches import autoencoder_collate
from legoml.utils.seed import worker_init_fn


def train_mnist_vae_v2(
    *,
    max_epochs: int = 5,
    device: str | None = None,
    latent_dim: int = 16,
    beta_end: float = 1.0,
    warmup_epochs: int = 3,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_ds = build_mnist(MNISTConfig(split="train", augmentation=False))
    val_ds = build_mnist(MNISTConfig(split="test", augmentation=False))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=autoencoder_collate, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=autoencoder_collate, worker_init_fn=worker_init_fn)

    # Model + Task
    model = MLPVAE(VAEConfig(input_shape=(1, 28, 28), latent_dim=latent_dim))
    task = VAEReconstructionTask(model=model, beta=0.0, recon_loss="bce", device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = None

    trainer = Trainer(
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        strategy=SingleDeviceStrategy(device=device, use_amp=False),
        callbacks=[
            TQDMProgressBar(leave=False),
            LearningRateMonitor(),
            ModelCheckpoint(save_dir="./models", run_name="mnist_vae_v2", every_n_epochs=1),
            LinearAttrWarmup(target=task, attr="beta", start=0.0, end=beta_end, warmup_epochs=warmup_epochs),
        ],
        max_epochs=max_epochs,
        log_every_n_steps=200,
    )

    return trainer.fit(train_loader, val_loader)

