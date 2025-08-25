import torch
from dataclasses import dataclass
from typing import Literal
from torch.utils.data import Dataset
from torchvision import datasets, transforms as T
from legoml.data.batches import ClassificationBatch, AutoencoderBatch


@dataclass
class CIFAR10Config:
    split: Literal["train", "test"] = "train"
    means: tuple[float, ...] = (0.49139968, 0.48215827, 0.44653124)
    stds: tuple[float, ...] = (0.24703233, 0.24348505, 0.26158768)
    augmentation: bool = False
    normalize: bool = True
    download: bool = True
    data_root: str = "./raw_data"


def build_cifar10(cfg: CIFAR10Config) -> Dataset:
    ts = []
    if cfg.split == "train" and cfg.augmentation:
        ts += [
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
    ts.append(T.ToTensor())
    if cfg.normalize:
        ts.append(T.Normalize(cfg.means, cfg.stds))

    return datasets.CIFAR10(
        root=cfg.data_root,
        train=(cfg.split == "train"),
        download=cfg.download,
        transform=T.Compose(ts),
    )


def classification_collate(batch: list[tuple[torch.Tensor, int]]):
    inputs = torch.stack([x for x, _ in batch], dim=0)
    targets = torch.tensor([y for _, y in batch], dtype=torch.int64)
    return ClassificationBatch(inputs=inputs, targets=targets)


def autoencoder_collate(batch: list[tuple[torch.Tensor, int]]):
    inputs = torch.stack([x for x, _ in batch], dim=0)
    return AutoencoderBatch(inputs=inputs)
