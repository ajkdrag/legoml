import torch.nn as nn


class Linear__LnAct(nn.Sequential):
    def __init__(
        self,
        c1,
        c2=None,
        dropout=0.0,
        act_fn=lambda: nn.ReLU(inplace=True),
    ):
        c2 = c2 or c1
        super().__init__(
            nn.Linear(c1, c2),
            nn.LayerNorm(c2),
            act_fn(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
