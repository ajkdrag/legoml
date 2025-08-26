import torch.nn as nn


class Conv__BnAct(nn.Sequential):
    def __init__(
        self,
        *,
        c1,
        c2,
        k=3,
        s=1,
        p=1,
        act_fn=lambda: nn.ReLU(inplace=True),
        g=1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels=c1,
                out_channels=c2,
                kernel_size=k,
                stride=s,
                padding=p,
                groups=g,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            act_fn(),
        )


class Conv_3x3__BnAct(Conv__BnAct):
    def __init__(
        self,
        *,
        c1,
        c2,
        act_fn=lambda: nn.ReLU(inplace=True),
        g=1,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            k=3,
            s=1,
            p=1,
            g=g,
            act_fn=act_fn,
        )


class Conv_1x1__BnAct(Conv__BnAct):
    def __init__(
        self,
        *,
        c1,
        c2,
        act_fn=lambda: nn.ReLU(inplace=True),
        g=1,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            k=1,
            s=1,
            p=0,
            g=g,
            act_fn=act_fn,
        )


class Conv_3x3_Down__BnAct(Conv__BnAct):
    def __init__(
        self,
        *,
        c1,
        c2,
        act_fn=lambda: nn.ReLU(inplace=True),
        g=1,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            k=3,
            s=2,
            p=1,
            g=g,
            act_fn=act_fn,
        )


class Conv_1x1_Down__BnAct(Conv__BnAct):
    def __init__(
        self,
        *,
        c1,
        c2,
        act_fn=lambda: nn.ReLU(inplace=True),
        g=1,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            k=1,
            s=2,
            p=0,
            g=g,
            act_fn=act_fn,
        )


class Conv_3x3__BnAct__Pool(nn.Sequential):
    def __init__(
        self,
        *,
        c1,
        c2,
        act_fn=lambda: nn.ReLU(inplace=True),
        pool_fn=lambda: nn.MaxPool2d(2, ceil_mode=True),
    ):
        super().__init__(
            Conv_3x3__BnAct(c1=c1, c2=c2, act_fn=act_fn),
            pool_fn(),
        )


class DWConv_3x3__BnAct(Conv_3x3__BnAct):
    def __init__(
        self,
        *,
        c1,
        act_fn=lambda: nn.ReLU(inplace=True),
    ):
        super().__init__(c1=c1, c2=c1, act_fn=act_fn, g=c1)


class DWConv_3x3_Down__BnAct(Conv_3x3_Down__BnAct):
    def __init__(
        self,
        *,
        c1,
        act_fn=lambda: nn.ReLU(inplace=True),
    ):
        super().__init__(c1=c1, c2=c1, act_fn=act_fn, g=c1)


class DWSepConv_3x3__BnAct(nn.Sequential):
    def __init__(
        self,
        *,
        c1,
        c2,
        dw_act_fn=lambda: nn.ReLU(inplace=True),
        pw_act_fn=lambda: nn.Identity(),
    ):
        super().__init__(
            DWConv_3x3__BnAct(c1=c1, act_fn=dw_act_fn),
            Conv_1x1__BnAct(c1=c1, c2=c2, act_fn=pw_act_fn),
        )


class Shortcut(nn.Sequential):
    """If channel change use conv projection for shortcut, else Identity fn"""

    def __init__(self, *, c1, c2=None):
        c2 = c2 or c1
        layers = []
        if c1 != c2:
            layers.append(Conv_1x1__BnAct(c1=c1, c2=c2, act_fn=lambda: nn.Identity()))
        else:
            layers.append(nn.Identity())
        super().__init__(*layers)


class Shortcut_Down(Conv_1x1_Down__BnAct):
    """If downsampling use conv projection for shortcut"""

    def __init__(self, *, c1, c2=None):
        c2 = c2 or c1
        super().__init__(c1=c1, c2=c2, act_fn=lambda: nn.Identity())
