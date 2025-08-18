from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from legoml.core.loops.trainer import Trainer
from legoml.core.strategies.single_device import SingleDeviceStrategy
from legoml.core.callbacks.progress import TQDMProgressBar
from legoml.core.callbacks.lr_monitor import LearningRateMonitor
from legoml.core.callbacks.checkpoint import ModelCheckpoint

from legoml.nn.vae import MLPVAE, VAEConfig
from legoml.tasks.vae import VAEReconstructionTask
from legoml.nn.mlp import MLP, MLPConfig
from legoml.nn.composite import VAEMuEncoder, FrozenEncoderClassifier
from legoml.tasks.image_classification import ImageClassificationTask
from legoml.objectives.cross_entropy import CrossEntropyObjective
from legoml.metrics.multiclass import MultiClassAccuracy

from legoml.data.mnist import MNISTConfig, build_mnist
from legoml.data.batches import classification_collate, autoencoder_collate
from legoml.utils.seed import worker_init_fn


def pretrain_autoencoder_and_classify_v2(
    *,
    device: str | None = None,
    latent_dim: int = 16,
    pretrain_epochs: int = 3,
    cls_epochs: int = 3,
    head_hidden: list[int] | None = None,
    freeze_encoder: bool = True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Pretrain VAE for reconstruction
    train_ds = build_mnist(MNISTConfig(split="train"))
    val_ds = build_mnist(MNISTConfig(split="test"))
    ae_train = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=autoencoder_collate, worker_init_fn=worker_init_fn)
    ae_val = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=autoencoder_collate, worker_init_fn=worker_init_fn)

    vae = MLPVAE(VAEConfig(input_shape=(1, 28, 28), latent_dim=latent_dim))
    vae_task = VAEReconstructionTask(model=vae, beta=1.0, recon_loss="bce", device=device)
    vae_opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

    vae_trainer = Trainer(
        task=vae_task,
        optimizer=vae_opt,
        scheduler=None,
        strategy=SingleDeviceStrategy(device=device, use_amp=False),
        callbacks=[
            TQDMProgressBar(leave=False),
            LearningRateMonitor(),
            ModelCheckpoint(save_dir="./models", run_name="mnist_ae_pretrain_v2", every_n_epochs=pretrain_epochs),
        ],
        max_epochs=pretrain_epochs,
        log_every_n_steps=200,
    )
    vae_state = vae_trainer.fit(ae_train, ae_val)

    # 2) Build frozen encoder → classification head
    encoder = VAEMuEncoder(vae)
    zdim = latent_dim
    head = MLP(MLPConfig(in_dim=zdim, hidden=head_hidden or [256, 128], out_dim=10, dropout=0.1))
    clf = FrozenEncoderClassifier(encoder=encoder, head=head, freeze_encoder=freeze_encoder)

    # 3) Fine-tune on classification with frozen encoder
    cls_train = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=classification_collate, worker_init_fn=worker_init_fn)
    cls_val = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=classification_collate, worker_init_fn=worker_init_fn)

    clf_task = ImageClassificationTask(
        model=clf,
        objective=CrossEntropyObjective(),
        metrics=[MultiClassAccuracy(name="accuracy")],
        device=device,
    )

    # Optimize only head parameters if encoder is frozen
    learnable_params = (p for p in clf.parameters() if p.requires_grad)
    clf_opt = torch.optim.Adam(learnable_params, lr=1e-3, weight_decay=1e-4)
    clf_sched = torch.optim.lr_scheduler.CosineAnnealingLR(clf_opt, T_max=cls_epochs, eta_min=1e-6)

    clf_trainer = Trainer(
        task=clf_task,
        optimizer=clf_opt,
        scheduler=clf_sched,
        strategy=SingleDeviceStrategy(device=device, use_amp=False),
        callbacks=[
            TQDMProgressBar(leave=False),
            LearningRateMonitor(),
            ModelCheckpoint(save_dir="./models", run_name="mnist_ae_cls_v2", every_n_epochs=1),
        ],
        max_epochs=cls_epochs,
        log_every_n_steps=100,
    )

    clf_state = clf_trainer.fit(cls_train, cls_val)
    return {"pretrain": vae_state, "finetune": clf_state}

