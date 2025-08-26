import torch.nn as nn
from legoml.nn.common import Conv_3x3__BnAct, Conv_3x3__BnAct__Pool
from legoml.nn.convs import InceptionResnet, ResBottleneck, ResBottleneck_Down
from legoml.nn.mlp import Linear__LnAct


class CNN__MLP_tiny_28x28(nn.Sequential):
    def __init__(self, c1=1):
        super().__init__(
            Conv_3x3__BnAct__Pool(c1=c1, c2=32),  # [32, 14, 14]
            Conv_3x3__BnAct(c1=32, c2=32),  # [32, 14, 14]
            Conv_3x3__BnAct__Pool(c1=32, c2=64),  # [64, 7, 7]
            Conv_3x3__BnAct(c1=64, c2=64),  # [64, 7, 7]
            nn.Flatten(),  # [64 * 7 * 7]
            Linear__LnAct(64 * 7 * 7, 128),  # [128]
            Linear__LnAct(128, 10),  # [10]
        )


class CNN__MLP_tiny_32x32(nn.Sequential):
    def __init__(self, c1=1):
        super().__init__(
            Conv_3x3__BnAct__Pool(c1=c1, c2=32),  # [32, 16, 16]
            Conv_3x3__BnAct(c1=32, c2=32),  # [32, 16, 16]
            Conv_3x3__BnAct__Pool(c1=32, c2=64),  # [64, 8, 8]
            Conv_3x3__BnAct(c1=64, c2=64),  # [64, 8, 8]
            nn.Flatten(),  # [64 * 8 * 8]
            Linear__LnAct(64 * 8 * 8, 64),  # [64]
            Linear__LnAct(64, 10),  # [10]
        )


class Resnet_style_32x32(nn.Sequential):
    def __init__(self, c1=1):
        super().__init__(
            Conv_3x3__BnAct__Pool(c1=c1, c2=32),  # [32, 16, 16]
            ResBottleneck(c1=32, c2=64),  # [64, 16, 16]
            ResBottleneck_Down(c1=64, c2=64),  # [64, 8, 8]
            ResBottleneck(c1=64, c2=128),  # [128, 8, 8]
            ResBottleneck_Down(c1=128, c2=128),  # [128, 4, 4]
            nn.Flatten(),  # [128 * 4 * 4]
            Linear__LnAct(128 * 4 * 4, 64),  # [64]
            Linear__LnAct(64, 10),  # [10]
        )


class InceptionResnet_style_32x32(nn.Sequential):
    def __init__(self, c1=1):
        super().__init__(
            ResBottleneck_Down(c1=c1, c2=32),  # [32, 16, 16]
            InceptionResnet(c1=32),  # [32, 16, 16]
            ResBottleneck_Down(c1=32, c2=64),  # [64, 8, 8]
            InceptionResnet(c1=64),  # [64, 8, 8]
            ResBottleneck_Down(c1=64, c2=128),  # [128, 4, 4]
            nn.Flatten(),  # [128 * 4 * 4]
            Linear__LnAct(128 * 4 * 4, 64),  # [64]
            Linear__LnAct(64, 10),  # [10]
        )


class AEBackbone__MLP(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            Linear__LnAct(128, 64),
            Linear__LnAct(64, 10),
        )

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)  # [128]
        x = self.head(x)  # [10]
        return x
