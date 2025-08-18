from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from legoml.core.loops.trainer import Trainer
from legoml.core.strategies.single_device import SingleDeviceStrategy
from legoml.core.callbacks.progress import TQDMProgressBar
from legoml.core.callbacks.lr_monitor import LearningRateMonitor
from legoml.core.callbacks.checkpoint import ModelCheckpoint
from legoml.metrics.binary import BinaryAccuracy
from legoml.metrics.f1 import BinaryF1
from legoml.tasks.multimodal_pair_similarity import PairSimilarityTask
from legoml.embedders.random import RandomTextEmbedder, RandomImageEmbedder
from legoml.nn.mlp import MLP, MLPConfig
from legoml.data.pair_dataset import PairCsvDataset
from legoml.data.batches import pair_collate
from legoml.utils.seed import worker_init_fn


def train_pair_similarity_v2(
    csv_path: str,
    image_root: str | None = None,
    *,
    text_dim: int | None = None,
    image_dim: int | None = None,
    hidden: list[int] | None = None,
    batch_size: int = 16,
    max_epochs: int = 1,
    device: str | None = None,
):
    """
    Multimodal pair similarity training using V2 components.

    To use HF encoders, replace Random*Embedder with wrappers that implement the same Protocol.
    The dataset CSV must have columns: text1,image1,text2,image2,label
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = PairCsvDataset(csv_path=csv_path, image_root=image_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=pair_collate, worker_init_fn=worker_init_fn)
    vl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=pair_collate, worker_init_fn=worker_init_fn)

    # Embedders (replace with HF-backed implementations)
    # If dims are provided, use them; else take from embedder after init
    text_emb = RandomTextEmbedder(dim=text_dim or 64)
    img_emb = RandomImageEmbedder(dim=image_dim or 64)

    in_dim = 2 * (text_emb.dim + img_emb.dim)  # concat z1 and z2
    clf = MLP(MLPConfig(in_dim=in_dim, hidden=hidden or [256, 128], out_dim=1, dropout=0.1))

    task = PairSimilarityTask(
        text_embedder=text_emb,
        image_embedder=img_emb,
        classifier=clf,
        metrics=[BinaryAccuracy(name="accuracy"), BinaryF1(name="f1")],
        device=device,
    )

    optimizer = torch.optim.Adam(task.model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = None

    trainer = Trainer(
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        strategy=SingleDeviceStrategy(device=device, use_amp=False),
        callbacks=[
            TQDMProgressBar(leave=False),
            LearningRateMonitor(),
            ModelCheckpoint(save_dir="./models", run_name="pair_sim_v2", every_n_epochs=1),
        ],
        max_epochs=max_epochs,
        log_every_n_steps=50,
    )

    return trainer.fit(dl, vl)
