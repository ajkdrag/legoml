from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
from torch.utils.data.dataloader import DataLoader

from experiments.data_utils import create_dataloaders
from experiments.image_clf.config import Config
from experiments.image_clf.models import (
    ConvNeXt_tiny_32x32,
)
from experiments.image_clf.steps import eval_step, train_step
from legoml.callbacks.checkpoint import CheckpointCallback
from legoml.callbacks.eval import EvalOnEpochEndCallback
from legoml.callbacks.metric import MetricsCallback
from legoml.core.context import Context
from legoml.core.engine import Engine
from legoml.metrics.multiclass import MultiClassAccuracy
from legoml.schedulers.reducelr_with_warmup import WarmupReduceLROnPlateau
from legoml.utils.log import get_logger
from legoml.utils.optim import default_groups, print_param_groups
from legoml.utils.seed import set_seed
from legoml.utils.summary import summarize_model
from legoml.utils.track import run

logger = get_logger(__name__)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
logger.info("Using device: %s", device.type)
set_seed(42)
config = Config(train_augmentation=True, max_epochs=30, train_bs=128)


def build_optim_and_sched(
    config: Config,
    model: nn.Module,
    train_dl: DataLoader,
) -> tuple[torch.optim.Optimizer, lrs.LRScheduler]:
    base_max_lr = 0.1 * (config.train_bs / 128)
    groups = default_groups(model, lr=base_max_lr, weight_decay=5e-4)

    base_lrs = [g["lr"] for g in groups]
    print_param_groups(model, groups)

    optimizer = torch.optim.SGD(
        groups,
        momentum=0.9,
        nesterov=True,
    )

    # scheduler = lrs.OneCycleLR(
    #     optimizer,
    #     max_lr=max_lrs,
    #     epochs=int(pct_phase_1 * config.max_epochs),
    #     steps_per_epoch=len(train_dl),
    #     pct_start=0.1,
    #     anneal_strategy="cos",
    # )

    scheduler = WarmupReduceLROnPlateau(
        optimizer,
        max_epochs=config.max_epochs,
        warmup_epochs=config.max_epochs // 10,
        max_lr=0.1,
        init_lr=0.0001,
        scheduler=lrs.ReduceLROnPlateau(optimizer),
    )

    return optimizer, scheduler


train_dl, eval_dl = create_dataloaders("cifar10", config, "classification")
model = ConvNeXt_tiny_32x32()
summary = summarize_model(model, next(iter(train_dl)).inputs, depth=2)

model.to(device)
optim, sched = build_optim_and_sched(config, model, train_dl)

with run(base_dir=Path("runs").joinpath("train_img_clf_cifar10")) as sess:
    train_context = Context(
        config=config,
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.05),
        optimizer=optim,
        scheduler=sched,
        device=device,
        scaler=torch.GradScaler(device=device.type),  # slow on M1 air
    )
    trainer = Engine(train_step, train_context)

    eval_context = Context(
        config=config,
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=device,
    )
    evaluator = Engine(
        eval_step,
        eval_context,
        callbacks=[
            MetricsCallback(metrics=[MultiClassAccuracy("eval_acc")]),
        ],
    )

    trainer.callbacks.extend(
        [
            EvalOnEpochEndCallback(evaluator, eval_dl, 1),
            MetricsCallback(metrics=[MultiClassAccuracy("train_acc")]),
            CheckpointCallback(
                dirpath=sess.get_artifact_dir().joinpath("checkpoints"),
                save_every_n_epochs=9999,
                save_on_engine_end=True,
                best_fn=lambda: evaluator.state.metrics["eval_acc"],
            ),
        ]
    )

    trainer.loop(train_dl, max_epochs=config.max_epochs)
    sess.log_params({"exp_config": asdict(config)})
    sess.log_text("model", f"{summary}\n\n{model}")
    sess.log_params({"trainer": trainer.to_dict()})
    sess.log_params({"evaluator": evaluator.to_dict()})
