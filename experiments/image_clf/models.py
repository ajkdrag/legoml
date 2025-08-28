from collections import OrderedDict
import torchinfo
import torch
import torch.nn as nn
from legoml.nn.layers import Conv3x3, GlobalAvgPool2d, LinearLayer, identity
from legoml.nn.blocks.conv import (
    ResNetBottleneck,
    ResNeXtBottleneck,
)


class CNN__MLP_tiny_32x32(nn.Sequential):
    def __init__(self, c1=3):
        stem = nn.Sequential(
            Conv3x3(c1=c1, c2=32),  # [32, 32, 32]
            nn.MaxPool2d(2, 2),  # [32, 16, 16]
        )
        backbone = nn.Sequential(
            Conv3x3(c1=32, c2=64),  # [64, 16, 16]
            nn.MaxPool2d(2, 2),  # [64, 8, 8]
            Conv3x3(c1=64, c2=64),  # [64, 8, 8]
            Conv3x3(c1=64, c2=64),  # [64, 8, 8]
            Conv3x3(c1=64, c2=64),  # [64, 8, 8]
        )
        head = nn.Sequential(
            GlobalAvgPool2d(),  # [64]
            LinearLayer(c1=64, c2=10, act=identity),
        )

        super().__init__(
            stem,
            backbone,
            head,
        )


class ResNet_tiny_32x32(nn.Sequential):
    def __init__(self, c1=3):
        stem = nn.Sequential(
            Conv3x3(c1=c1, c2=32),  # [32, 32, 32]
            nn.MaxPool2d(2, 2),  # [32, 16, 16]
        )
        backbone = nn.Sequential(
            ResNetBottleneck(c1=32, c2=64, f=2, pre_normact=True),  # [64, 16, 16]
            nn.MaxPool2d(2, 2),  # [64, 8, 8]
            ResNetBottleneck(c1=64, c2=64, f=2, pre_normact=True),  # [64, 8, 8]
            ResNetBottleneck(
                c1=64, c2=64, f=2, pre_normact=True, drop_path=0.1
            ),  # [64, 8, 8]
            ResNetBottleneck(
                c1=64, c2=64, f=2, pre_normact=True, drop_path=0.1
            ),  # [64, 8, 8]
        )
        head = nn.Sequential(
            GlobalAvgPool2d(),  # [64]
            nn.ReLU(inplace=True),
            LinearLayer(c1=64, c2=10, act=identity),
        )

        super().__init__(
            stem,
            backbone,
            head,
        )


# class MB_style_32x32(nn.Sequential):
#     def __init__(self, c1=3):
#         super().__init__(
#             Conv_3x3__BnAct(c1=c1, c2=32),  # [32, 32, 32]
#             nn.Sequential(
#                 MBConv(c1=32, c2=32),  # [32, 32, 32]
#                 MBConv(c1=32, c2=32),  # [32, 32, 32]
#             ),
#             nn.Sequential(
#                 MBConv_Down(c1=32, c2=64),  # [64, 16, 16]
#                 MBConv(c1=64, c2=64),  # [64, 16, 16]
#             ),
#             nn.Sequential(
#                 MBConv_Down(c1=64, c2=128),  # [128, 8, 8]
#                 MBConv(c1=128, c2=128),  # [128, 8, 8]
#             ),
#             nn.AdaptiveAvgPool2d((1, 1)),  # [128, 1, 1]
#             nn.Flatten(),  # [128]
#             Linear__LnAct(128 * 1 * 1, 64),  # [64]
#             nn.Linear(64, 10),
#         )


class AEBackbone__MLP(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            LinearLayer(c1=128, c2=64),
            LinearLayer(c1=64, c2=10),
        )

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)  # [128]
        x = self.head(x)  # [10]
        return x


class PreActBasic(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.proj = None
        if stride != 1 or c_in != c_out:
            self.proj = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False)

    def forward(self, x):
        y = self.conv1(self.relu1(self.bn1(x)))
        y = self.conv2(self.relu2(self.bn2(y)))
        skip = x if self.proj is None else self.proj(x)
        return y + skip


class ResNet_CIFAR(nn.Module):
    def __init__(self, num_blocks=(3, 3, 3), widths=(16, 32, 64), num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)  # no pool
        self.stage1 = self._make_stage(widths[0], widths[0], num_blocks[0], stride=1)
        self.stage2 = self._make_stage(widths[0], widths[1], num_blocks[1], stride=2)
        self.stage3 = self._make_stage(widths[1], widths[2], num_blocks[2], stride=2)
        self.bn = nn.BatchNorm2d(widths[2])
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(widths[2], num_classes)

    def _make_stage(self, c_in, c_out, n, stride):
        blocks = [PreActBasic(c_in, c_out, stride)]
        for _ in range(n - 1):
            blocks.append(PreActBasic(c_out, c_out, 1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.relu(self.bn(x))
        x = x.mean(dim=(2, 3))  # global avg pool
        return self.head(x)


if __name__ == "__main__":
    dummy_ip = torch.randn(1, 3, 32, 32)
    model = ResNet_tiny_32x32()
    torchinfo.summary(
        model,
        input_data=dummy_ip,
        row_settings=["hide_recursive_layers"],
    )
    print(model)
