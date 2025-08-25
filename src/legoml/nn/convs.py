import torch.nn as nn


class Conv_3x3__BnAct(nn.Sequential):
    def __init__(
        self,
        c1,
        c2,
        act_fn=lambda: nn.ReLU(inplace=True),
    ):
        super().__init__(
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


class Conv_1x1__BnAct(nn.Sequential):
    def __init__(
        self,
        c1,
        c2,
        act_fn=lambda: nn.ReLU(inplace=True),
    ):
        super().__init__(
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


class Conv_3x3_Down__BnAct(nn.Sequential):
    def __init__(
        self,
        c1,
        c2,
        act_fn=lambda: nn.ReLU(inplace=True),
    ):
        super().__init__(
            nn.Conv2d(
                c1,
                c2,
                3,
                2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            act_fn(),
        )


class Conv_1x1_Down__BnAct(nn.Sequential):
    def __init__(
        self,
        c1,
        c2,
        act_fn=lambda: nn.ReLU(inplace=True),
    ):
        super().__init__(
            nn.Conv2d(
                c1,
                c2,
                1,
                2,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            act_fn(),
        )


class Conv_3x3__BnAct__Pool(nn.Sequential):
    def __init__(
        self,
        c1,
        c2,
        act_fn=lambda: nn.ReLU(inplace=True),
        pool_fn=lambda: nn.MaxPool2d(2, ceil_mode=True),
    ):
        super().__init__(
            Conv_3x3__BnAct(c1, c2, act_fn),
            pool_fn(),
        )


class DWConv_3x3__BnAct(nn.Sequential):
    def __init__(
        self,
        c1,
        act_fn=lambda: nn.ReLU(inplace=True),
    ):
        super().__init__(
            nn.Conv2d(
                c1,
                c1,
                3,
                1,
                padding=1,
                groups=c1,
                bias=False,
            ),
            nn.BatchNorm2d(c1),
            act_fn(),
        )


class DWSepConv_3x3__BnAct(nn.Sequential):
    def __init__(
        self,
        c1,
        c2,
        dw_act_fn=lambda: nn.ReLU(inplace=True),
        pw_act_fn=lambda: nn.Identity(),
    ):
        super().__init__(
            DWConv_3x3__BnAct(c1, dw_act_fn),
            Conv_1x1__BnAct(c1, c2, pw_act_fn),
        )


class Shortcut(nn.Sequential):
    def __init__(self, c1, c2=None):
        c2 = c2 or c1
        layers = []
        if c1 != c2:
            layers.append(Conv_1x1__BnAct(c1, c2))
        else:
            layers.append(nn.Identity())
        super().__init__(*layers)


class Shortcut_Down(nn.Sequential):
    def __init__(self, c1, c2=None):
        c2 = c2 or c1
        layers = []
        if c1 != c2:
            layers.append(Conv_1x1_Down__BnAct(c1, c2))
        else:
            layers.append(nn.AvgPool2d(2, ceil_mode=True))

        super().__init__(*layers)
