import enum


class Mode(enum.Enum):
    TRAIN = "train"
    EVAL = "eval"


class Device(enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
