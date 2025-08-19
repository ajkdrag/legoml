from enum import Enum


class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"


class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

