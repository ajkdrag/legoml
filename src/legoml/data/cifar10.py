from dataclasses import dataclass
from typing import Literal

import albumentations as A
import torch
from albumentations.core.composition import TransformsSeqType
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


def get_train_tfms(cfg: CIFAR10Config) -> TransformsSeqType:
    return [
        A.RandomResizedCrop((32, 32), (0.75, 1), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(0.2, 0.1, 10, fill=127, p=0.5),
        A.ColorJitter(0.2, 0.3, 0.2, 0.1, p=1.0),
        A.CoarseDropout(
            fill=127, hole_height_range=(8, 12), hole_width_range=(8, 12), p=0.2
        ),
    ]


def get_train_tfms_v2(cfg: CIFAR10Config) -> TransformsSeqType:
    return [
        A.RandomResizedCrop((32, 32), (0.75, 1), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.2, 0.3, 0.2, 0.02, p=0.6),
        A.Affine(0.2, 0.1, rotate=10, fill=0, p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(8, 12),
            hole_width_range=(8, 12),
            fill=0,  # zero out
            p=0.2,
        ),
    ]


def get_eval_tfms(cfg: CIFAR10Config) -> TransformsSeqType:
    return []


def get_essential_tfms(cfg: CIFAR10Config) -> TransformsSeqType:
    return [
        A.Normalize(mean=cfg.means, std=cfg.stds),
        A.ToTensorV2(),
    ]


class Cifar10Dataset(datasets.CIFAR10):
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
    tfms = []
    if cfg.split == "train" and cfg.augment:
        tfms += get_train_tfms_v2(cfg)
    elif cfg.split == "test" and cfg.augment:
        tfms += get_eval_tfms(cfg)

    tfms += get_essential_tfms(cfg)
    logger.info("Using transforms: %s", tfms)

    return Cifar10Dataset(
        root=cfg.data_root,
        train=(cfg.split == "train"),
        download=cfg.download,
        transform=A.Compose(tfms),
    )


def classification_collate(batch: list[tuple[torch.Tensor, int]]):
    inputs = torch.stack([x for x, _ in batch], dim=0)
    targets = torch.tensor([y for _, y in batch], dtype=torch.int64)
    return ClassificationBatch(inputs=inputs, targets=targets)


def autoencoder_collate(batch: list[tuple[torch.Tensor, int]]):
    inputs = torch.stack([x for x, _ in batch], dim=0)
    return AutoencoderBatch(inputs=inputs)
