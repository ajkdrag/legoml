from torch.utils.data.dataloader import DataLoader
from legoml.utils.logging import get_logger

import legoml.data.mnist as mnist
import legoml.data.cifar10 as cifar10
from experiments.image_clf.config import Config

logger = get_logger(__name__)


def get_mnist_dls(config: Config) -> tuple[DataLoader, ...]:
    train_ds = mnist.build_mnist(
        mnist.MNISTConfig(
            split="train",
            augmentation=config.train_augmentation,
            data_root=config.data_root,
        )
    )

    val_ds = mnist.build_mnist(
        mnist.MNISTConfig(
            split="test",
            augmentation=False,
            data_root=config.data_root,
        )
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train_bs,
        shuffle=True,
        collate_fn=mnist.classification_collate,
    )
    eval_loader = DataLoader(
        val_ds,
        batch_size=config.eval_bs,
        shuffle=False,
        collate_fn=mnist.classification_collate,
    )
    logger.info(
        "Created data loaders",
        train_size=len(train_loader),
        val_size=len(eval_loader),
    )
    return train_loader, eval_loader


def get_cifar10_dls(config: Config) -> tuple[DataLoader, ...]:
    train_ds = cifar10.build_cifar10(
        cifar10.CIFAR10Config(
            split="train",
            augmentation=config.train_augmentation,
            data_root=config.data_root,
        )
    )

    val_ds = cifar10.build_cifar10(
        cifar10.CIFAR10Config(
            split="test",
            augmentation=False,
            data_root=config.data_root,
        )
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train_bs,
        shuffle=True,
        collate_fn=cifar10.classification_collate,
    )
    eval_loader = DataLoader(
        val_ds,
        batch_size=config.eval_bs,
        shuffle=False,
        collate_fn=cifar10.classification_collate,
    )
    logger.info(
        "Created data loaders",
        train_size=len(train_loader),
        val_size=len(eval_loader),
    )
    return train_loader, eval_loader
