import torch.nn as nn

relu = nn.ReLU(inplace=True)
gelu = nn.GELU()
relu6 = nn.ReLU6(inplace=True)
silu = nn.SiLU()


def gelu_fn():
    return gelu


def relu_fn():
    return relu


def relu6_fn():
    return relu6


def silu_fn():
    return silu
