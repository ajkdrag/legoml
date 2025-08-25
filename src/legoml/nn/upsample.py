import torch.nn as nn


class ConvTranspose_3x3__BnAct(nn.Sequential):
    def __init__(
        self,
        c1,
        c2,
        op_pad=0,
        act_fn=lambda: nn.ReLU(inplace=True),
    ):
        super().__init__(
            nn.ConvTranspose2d(
                c1,
                c2,
                3,
                1,
                padding=1,
                output_padding=op_pad,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            act_fn(),
        )


class ConvTranspose_3x3_Up__BnAct(nn.Sequential):
    def __init__(
        self,
        c1,
        c2,
        op_pad=1,
        act_fn=lambda: nn.ReLU(inplace=True),
    ):
        super().__init__(
            nn.ConvTranspose2d(
                c1,
                c2,
                3,
                2,
                padding=1,
                output_padding=op_pad,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            act_fn(),
        )


class ConvTranspose_1x1__BnAct(nn.Sequential):
    def __init__(
        self,
        c1,
        c2,
        op_pad=0,
        act_fn=lambda: nn.ReLU(inplace=True),
    ):
        super().__init__(
            nn.ConvTranspose2d(
                c1,
                c2,
                1,
                1,
                padding=0,
                output_padding=op_pad,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            act_fn(),
        )


class Upsample_Conv_3x3__BnAct(nn.Sequential):
    def __init__(
        self,
        c1,
        c2,
        act_fn=lambda: nn.ReLU(inplace=True),
        mode="nearest",
    ):
        super().__init__(
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(
                c1,
                c2,
                3,
                1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            act_fn(),
        )


class Upsample_Conv_1x1__BnAct(nn.Sequential):
    def __init__(
        self,
        c1,
        c2,
        act_fn=lambda: nn.ReLU(inplace=True),
        mode="nearest",
    ):
        super().__init__(
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(
                c1,
                c2,
                1,
                1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            act_fn(),
        )


class Shortcut_Up(nn.Sequential):
    def __init__(self, c1, c2=None, upsample_mode="nearest"):
        c2 = c2 or c1
        layers = []
        if c1 != c2:
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode=upsample_mode),
                    nn.Conv2d(
                        c1,
                        c2,
                        1,
                        1,
                        padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(c2),
                ]
            )
        else:
            layers.append(nn.Upsample(scale_factor=2, mode=upsample_mode))

        super().__init__(*layers)

