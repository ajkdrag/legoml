import torch.nn as nn
from einops.layers.torch import Rearrange
from legoml.nn.convs import Conv_3x3__BnAct, Conv_3x3__BnAct__Pool
from legoml.nn.upsample import ConvTranspose_3x3_Up__BnAct
from legoml.nn.mlp import Linear__LnAct


class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            Conv_3x3__BnAct__Pool(1, 32),  # [32, 14, 14]
            Conv_3x3__BnAct(32, 32),  # [32, 14, 14]
            Conv_3x3__BnAct__Pool(32, 64),  # [64, 7, 7]
            Conv_3x3__BnAct(64, 64),  # [64, 7, 7]
            Conv_3x3__BnAct__Pool(64, 128),  # [128, 4, 4]
            nn.Flatten(),  # [128 * 4 * 4]
            Linear__LnAct(128 * 4 * 4, 128),  # [128]
        )


class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            Linear__LnAct(128, 128 * 4 * 4),  # [128 * 4 * 4]
            Rearrange(
                "b (c h w) -> b c h w",
                c=128,
                h=4,
                w=4,
            ),  # [128, 4, 4]
            ConvTranspose_3x3_Up__BnAct(128, 64, op_pad=0),  # [64, 7, 7]
            Conv_3x3__BnAct(64, 64),  # [64, 7, 7]
            ConvTranspose_3x3_Up__BnAct(64, 32),  # [32, 14, 14]
            Conv_3x3__BnAct(32, 32),  # [32, 14, 14]
            ConvTranspose_3x3_Up__BnAct(32, 1),  # [1, 28, 28]
        )


class Autoencoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            Encoder(),
            Decoder(),
        )
