import torch.nn as nn
from legoml.nn.convs import Conv_3x3__BnAct, Conv_3x3__BnAct__Pool
from legoml.nn.mlp import Linear__LnAct


class CNN__MLP_tiny_28x28(nn.Sequential):
    def __init__(self):
        super().__init__(
            Conv_3x3__BnAct__Pool(1, 32),  # [32, 14, 14]
            Conv_3x3__BnAct(32, 32),  # [32, 14, 14]
            Conv_3x3__BnAct__Pool(32, 64),  # [64, 7, 7]
            Conv_3x3__BnAct(64, 64),  # [64, 7, 7]
            nn.Flatten(),  # [64 * 7 * 7]
            Linear__LnAct(64 * 7 * 7, 128),  # [128]
            Linear__LnAct(128, 10),  # [10]
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
