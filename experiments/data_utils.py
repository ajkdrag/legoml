from torch.utils.data.dataloader import DataLoader

import legoml.data.cifar10 as cifar10
import legoml.data.mnist as mnist
from legoml.utils.logging import get_logger

logger = get_logger(__name__)


def create_mnist_loaders(
    config, task_type: str = "classification"
) -> tuple[DataLoader, ...]:
    """Create MNIST train/val dataloaders for different tasks.

    Args:
        config: Config object with train_bs, eval_bs, train_augmentation, data_root
        task_type: "classification" or "autoencoder"
    """
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

    # Select collate function based on task type
    collate_fn = {
        "classification": mnist.classification_collate,
        "autoencoder": mnist.autoencoder_collate,
    }[task_type]

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train_bs,
        shuffle=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        val_ds,
        batch_size=config.eval_bs,
        shuffle=False,
        collate_fn=collate_fn,
    )

    logger.info(
        "Created MNIST data loaders",
        task_type=task_type,
        train_size=len(train_loader),
        val_size=len(eval_loader),
    )
    return train_loader, eval_loader


def create_cifar10_loaders(
    config, task_type: str = "classification"
) -> tuple[DataLoader, ...]:
    """Create CIFAR10 train/val dataloaders for different tasks.

    Args:
        config: Config object with train_bs, eval_bs, train_augmentation, data_root
        task_type: "classification" or "autoencoder"
    """
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

    # Select collate function based on task type
    collate_fn = {
        "classification": cifar10.classification_collate,
        "autoencoder": cifar10.autoencoder_collate,
    }[task_type]

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train_bs,
        shuffle=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        val_ds,
        batch_size=config.eval_bs,
        shuffle=False,
        collate_fn=collate_fn,
    )

    logger.info(
        "Created CIFAR10 data loaders",
        task_type=task_type,
        train_size=len(train_loader),
        val_size=len(eval_loader),
    )
    return train_loader, eval_loader


# Registry for easy dataset selection
DATASET_FACTORIES = {
    "mnist": create_mnist_loaders,
    "cifar10": create_cifar10_loaders,
}


def create_dataloaders(
    dataset_name: str, config, task_type: str = "classification"
) -> tuple[DataLoader, ...]:
    """Create dataloaders for any supported dataset.

    Args:
        dataset_name: "mnist" or "cifar10"
        config: Config object with required fields
        task_type: "classification" or "autoencoder"
    """
    if dataset_name not in DATASET_FACTORIES:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Available: {list(DATASET_FACTORIES.keys())}"
        )

    return DATASET_FACTORIES[dataset_name](config, task_type)

