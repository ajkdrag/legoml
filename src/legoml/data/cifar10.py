from dataclasses import dataclass
from typing import Literal

import albumentations as A
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets

from legoml.data.batches import AutoencoderBatch, ClassificationBatch
from legoml.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class CIFAR10Config:
    split: Literal["train", "test"] = "train"
    means: tuple[float, ...] = (0.49139968, 0.48215827, 0.44653124)
    stds: tuple[float, ...] = (0.24703233, 0.24348505, 0.26158768)
    augment: bool = False
    download: bool = True
    data_root: str = "./raw_data"


def get_train_tfms_alb(cfg: CIFAR10Config):
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=36,
                min_width=36,
                border_mode=0,
                fill=tuple(int(i * 255) for i in cfg.means),
            ),
            A.RandomCrop(32, 32),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0, 0.08),
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.7,
            ),
            A.OneOf(
                [
                    A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0),
                    A.RandomBrightnessContrast(0.2, 0.2, p=1.0),
                    A.HueSaturationValue(5, 20, 10, p=1.0),
                ],
                p=0.7,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_width_range=(1, 8),
                hole_height_range=(1, 8),
                fill=tuple(int(i * 255) for i in cfg.means),
                p=0.5,
            ),
            A.Normalize(mean=cfg.means, std=cfg.stds),
            A.ToTensorV2(),
        ]
    )


def get_train_tfms_tv(cfg: CIFAR10Config):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=12),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(cfg.means, cfg.stds),
            transforms.RandomErasing(p=0.2),
        ]
    )


def get_eval_tfms_tv(cfg: CIFAR10Config):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cfg.means, cfg.stds),
        ]
    )


def get_eval_tfms_alb(cfg: CIFAR10Config):
    return A.Compose(
        [
            A.Normalize(mean=cfg.means, std=cfg.stds),
            A.ToTensorV2(),
        ]
    )


class Cifar10AlbumentationsDataset(datasets.CIFAR10):
    def __init__(
        self, root="~/data/cifar10", train=True, download=True, transform=None
    ):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def build_cifar10(cfg: CIFAR10Config) -> Dataset:
    tfms_fn = (
        get_eval_tfms_tv
        if (not cfg.augment or cfg.split == "test")
        else get_train_tfms_tv
    )

    tfms = tfms_fn(cfg)

    logger.info("Using transforms: %s", tfms)

    return datasets.CIFAR10(
        root=cfg.data_root,
        train=(cfg.split == "train"),
        download=cfg.download,
        transform=tfms,
    )


def classification_collate(batch: list[tuple[torch.Tensor, int]]):
    inputs = torch.stack([x for x, _ in batch], dim=0)
    targets = torch.tensor([y for _, y in batch], dtype=torch.int64)
    return ClassificationBatch(inputs=inputs, targets=targets)


def autoencoder_collate(batch: list[tuple[torch.Tensor, int]]):
    inputs = torch.stack([x for x, _ in batch], dim=0)
    return AutoencoderBatch(inputs=inputs)
