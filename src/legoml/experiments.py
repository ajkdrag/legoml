import torch
from dataclasses import field
from typing import List
from torch.utils.data import DataLoader

from legoml.core.constants import Device
from legoml.core.contracts.metric import Metric
from legoml.core.loops.trainer import Trainer
from legoml.core.strategies.single_device import SingleDeviceStrategy
from legoml.core.callbacks.progress import TQDMProgressBar
from legoml.core.callbacks.checkpoint import ModelCheckpoint
from legoml.core.callbacks.lr_monitor import LearningRateMonitor
from legoml.objectives.cross_entropy import CrossEntropyObjective
from legoml.tasks.image_classification import ImageClassificationTask
from legoml.core.callbacks.interval import EpochIntervalHook
from legoml.metrics.multiclass import MultiClassAccuracy
from legoml.nn.tinycnn import TinyCNNNode
from legoml.nn.mlp import MLPNode
from legoml.nn.activations import ReluNode
from legoml.nn.base import NoopNode
from legoml.data.mnist import MNISTConfig, build_mnist
from legoml.data.batches import classification_collate
from legoml.utils.seed import set_seed


def train_tiny_cnn_v2(max_epochs: int = 1):
    set_seed(42)
    device = Device.MPS

    node = TinyCNNNode(
        input_channels=1,
        mlp=MLPNode(
            dims=[128, 10],
            activation=ReluNode(),
            last_activation=NoopNode(),
        ),
    )
    model = node.build()

    # Data
    train_ds = build_mnist(MNISTConfig(split="train", augmentation=False))
    val_ds = build_mnist(MNISTConfig(split="test", augmentation=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=classification_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=classification_collate,
    )

    # Task + metrics
    metrics: List[Metric] = [MultiClassAccuracy(name="accuracy")]
    task = ImageClassificationTask(
        model=model,
        objective=CrossEntropyObjective(),
        metrics=metrics,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )

    trainer = Trainer(
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        strategy=SingleDeviceStrategy(device=device, use_amp=False),
        callbacks=[
            TQDMProgressBar(leave=False),
            LearningRateMonitor(),
            ModelCheckpoint(
                save_dir="./models", run_name="tiny_cnn_v2", every_n_epochs=1
            ),
        ],
        max_epochs=max_epochs,
        log_every_n_steps=100,
    )

    return trainer.fit(train_loader, val_loader)


def train_tiny_cnn_v2_with_meta_hook(max_epochs: int = 2):
    """Same as train_tiny_cnn_v2, but runs a user hook every epoch."""
    set_seed(42)
    device = Device.MPS

    cfg = TinyCNNNode(
        num_classes=10,
    )
    model = cfg.build()

    train_loader = DataLoader(
        build_mnist(MNISTConfig(split="train")),
        batch_size=64,
        shuffle=True,
        collate_fn=classification_collate,
    )
    val_loader = DataLoader(
        build_mnist(MNISTConfig(split="test")),
        batch_size=64,
        shuffle=False,
        collate_fn=classification_collate,
    )

    task = ImageClassificationTask(
        model=model,
        objective=CrossEntropyObjective(),
        metrics=[MultiClassAccuracy(name="accuracy")],
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )

    def user_meta_hook(state):
        # usecase maybe meta-model training or evaluation routine
        print(f"Epoch {state.epoch} - User meta hook called")
        pass

    trainer = Trainer(
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        strategy=SingleDeviceStrategy(device=device, use_amp=False),
        callbacks=[
            TQDMProgressBar(leave=False),
            LearningRateMonitor(),
            ModelCheckpoint(
                save_dir="./models",
                run_name="tiny_cnn_v2_meta",
                every_n_epochs=1,
            ),
            EpochIntervalHook(
                every_n_epochs=1,
                fn=user_meta_hook,
                when="on_train_epoch_end",
            ),
        ],
        max_epochs=max_epochs,
        log_every_n_steps=100,
    )

    return trainer.fit(train_loader, val_loader)
