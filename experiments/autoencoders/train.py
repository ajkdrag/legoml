import torch
import torch.optim.lr_scheduler as lrs
import torchsummary
from pathlib import Path

from dataclasses import asdict
from experiments.autoencoders.config import Config
from experiments.data_utils import create_dataloaders
from experiments.autoencoders.steps import train_step
from experiments.autoencoders.models import Autoencoder
from legoml.callbacks.checkpoint import CheckpointCallback
from legoml.core.context import Context
from legoml.core.engine import Engine
from legoml.utils.logging import get_logger
from legoml.utils.seed import set_seed
from legoml.utils.track import run

logger = get_logger(__name__)
device = torch.device("mps")
set_seed(42)
config = Config(train_augmentation=False)


def build_optim_and_sched(
    config: Config, model: torch.nn.Module
) -> tuple[torch.optim.Optimizer, lrs.LRScheduler]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    scheduler = lrs.CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs,
        eta_min=1e-6,
    )
    return optimizer, scheduler


model = Autoencoder()
optim, sched = build_optim_and_sched(config, model)
train_dl, eval_dl = create_dataloaders("mnist", config, "autoencoder")

with run(base_dir=Path("runs").joinpath("autoencoder")) as sess:
    train_context = Context(
        config=config,
        model=model,
        loss_fn=torch.nn.MSELoss(),
        optimizer=optim,
        scheduler=sched,
        device=device,
        # scaler=torch.GradScaler(device=device.type), # slow on M1 air
    )
    trainer = Engine(train_step, train_context)

    trainer.callbacks.extend(
        [
            CheckpointCallback(
                dirpath=sess.get_artifact_dir().joinpath("checkpoints"),
                save_on_engine_end=True,
            ),
        ]
    )

    torchsummary.summary(
        model,
        next(iter(train_dl)).inputs[0].shape,
    )
    model.to(device)
    trainer.loop(train_dl, max_epochs=config.max_epochs)
    sess.log_params({"exp_config": asdict(config)})
    sess.log_text("model", str(model))
    sess.log_params({"trainer": trainer.to_dict()})
