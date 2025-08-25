from torch.utils.data.dataloader import DataLoader
from legoml.utils.logging import get_logger

from legoml.data.mnist import MNISTConfig, build_mnist, classification_collate
from experiments.image_clf.config import Config

logger = get_logger(__name__)


def get_dls(config: Config) -> tuple[DataLoader, ...]:
    train_ds = build_mnist(
        MNISTConfig(
            split="train",
            augmentation=config.train_augmentation,
            data_root=config.data_root,
        )
    )

    val_ds = build_mnist(
        MNISTConfig(
            split="test",
            augmentation=False,
            data_root=config.data_root,
        )
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train_bs,
        shuffle=True,
        collate_fn=classification_collate,
    )
    eval_loader = DataLoader(
        val_ds,
        batch_size=config.eval_bs,
        shuffle=False,
        collate_fn=classification_collate,
    )
    logger.info(
        "Created data loaders",
        train_size=len(train_loader),
        val_size=len(eval_loader),
    )
    return train_loader, eval_loader
