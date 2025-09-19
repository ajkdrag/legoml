from dataclasses import dataclass


@dataclass
class Config:
    train_bs: int = 64
    eval_bs: int = 32
    train_augmentation: bool = False
    max_epochs: int = 2
    train_log_interval: int = 100  # steps
    eval_log_interval: int = 100  # steps
    data_root: str = "./raw_data"
    num_workers: int = 0
    persistent_workers: bool = False
    eval_interval: int = 1  # epochs
    eval_ema_interval: int = 5  # epochs
