import os
import random

import numpy as np
import torch
from torch.utils.data import random_split, Dataset


def set_seed(seed: int = 666, precision: int = 10) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(precision=precision)


def _train_val_split(
    dataset, validation_split: float = 0.2, seed: int = 666
) -> tuple[Dataset, Dataset | None]:
    """
    Deterministic split using self.seed.
    Called by both train/val builders so they stay consistent.
    """
    if not (0.0 < validation_split < 1.0):
        return dataset, None

    n_total = len(dataset)
    n_train = int((1 - validation_split) * n_total)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_ds, val_ds
