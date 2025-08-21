from dataclasses import dataclass
from typing import Literal
from torch.utils.data import Dataset
from torchvision import datasets, transforms as T


@dataclass
class MNISTConfig:
    split: Literal["train", "test"] = "train"
    means: tuple[float, ...] = (0.1307,)
    stds: tuple[float, ...] = (0.3081,)
    augmentation: bool = False
    normalize: bool = True
    download: bool = True
    data_root: str = "./raw_data"


def build_mnist(cfg: MNISTConfig) -> Dataset:
    ts = []
    if cfg.split == "train" and cfg.augmentation:
        ts += [
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
    ts.append(T.ToTensor())
    if cfg.normalize:
        ts.append(T.Normalize(cfg.means, cfg.stds))

    return datasets.MNIST(
        root=cfg.data_root,
        train=(cfg.split == "train"),
        download=cfg.download,
        transform=T.Compose(ts),
    )
