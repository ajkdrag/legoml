from dataclasses import dataclass
from pathlib import Path
from torchvision import datasets, transforms as T

from legoml.core.base import DatasetForClassificationNode
from legoml.utils.logging import get_logger
from legoml.utils.misc import _train_val_split
from torch.utils.data import Dataset


logger = get_logger("dataset.mnist")


@dataclass
class MNISTNode(DatasetForClassificationNode):
    data_root: Path = Path("./data")
    num_classes: int = 10
    download: bool = True
    normalize: bool = True
    augmentation: bool = False  # applied to training only
    means: tuple[float, ...] = (0.1307,)
    stds: tuple[float, ...] = (0.3081,)
    val_split: float = 0.1

    def __post_init__(self):
        self.full_train = self._get_full_dataset(train=True)

    def _transforms(self, is_train: bool) -> T.Compose:
        ts = []
        if is_train and self.augmentation:
            ts += [
                T.RandomHorizontalFlip(),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
        ts.append(T.ToTensor())
        if self.normalize:
            ts.append(T.Normalize(self.means, self.stds))
        return T.Compose(ts)

    def _get_full_dataset(self, train: bool):
        return datasets.MNIST(
            root=self.data_root,
            train=train,
            download=self.download,
            transform=self._transforms(train),
        )

    def _get_train_split(self):
        split = _train_val_split(
            self.full_train,
            self.val_split,
            seed=self.seed,
        )[0]
        logger.info(f"MNIST train split: {len(split)} samples")
        return split

    def _get_test_split(self):
        split = self._get_full_dataset(train=False)
        logger.info(f"MNIST test split: {len(split)} samples")
        return split

    def _get_validation_split(self):
        split = _train_val_split(
            self.full_train,
            self.val_split,
            seed=self.seed,
        )[1]
        logger.info(f"MNIST validation split: {len(split)} samples")
        return split

    def build(self, split) -> Dataset:
        if split == "train":
            return self._get_train_split()
        elif split == "test":
            return self._get_test_split()
        elif split == "val":
            return self._get_validation_split()
        else:
            raise ValueError(f"Unknown split: {self.split}")
