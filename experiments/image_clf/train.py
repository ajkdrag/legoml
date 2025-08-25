from dataclasses import asdict
from pathlib import Path

import torch
import torch.optim.lr_scheduler as lrs
import torchsummary

from experiments.image_clf.config import Config
from experiments.image_clf.data import get_cifar10_dls
from experiments.image_clf.models import CNN__MLP_tiny_32x32
from experiments.image_clf.steps import eval_step, train_step
from legoml.callbacks.checkpoint import CheckpointCallback
from legoml.callbacks.eval import EvalOnEpochEndCallback
from legoml.callbacks.metric import MetricsCallback
from legoml.core.context import Context
from legoml.core.engine import Engine
from legoml.metrics.multiclass import MultiClassAccuracy
from legoml.utils.logging import get_logger
from legoml.utils.seed import set_seed
from legoml.utils.track import run

logger = get_logger(__name__)
device = torch.device("mps")
set_seed(42)
config = Config(train_augmentation=True, max_epochs=5)


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


train_dl, eval_dl = get_cifar10_dls(config)
model = CNN__MLP_tiny_32x32(c1=3)
optim, sched = build_optim_and_sched(config, model)

with run(base_dir=Path("runs").joinpath("train_img_clf_cifar10")) as sess:
    train_context = Context(
        config=config,
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optim,
        scheduler=sched,
        device=device,
        # scaler=torch.GradScaler(device=device.type), # slow on M1 air
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
            MetricsCallback(metrics=[MultiClassAccuracy("eval_accuracy")]),
        ],
    )

    trainer.callbacks.extend(
        [
            EvalOnEpochEndCallback(evaluator, eval_dl, 1),
            MetricsCallback(metrics=[MultiClassAccuracy("train_accuracy")]),
            CheckpointCallback(
                dirpath=sess.get_artifact_dir().joinpath("checkpoints"),
                save_every_n_epochs=9999,
                save_on_engine_end=True,
                best_fn=lambda: evaluator.state.metrics["eval_accuracy"],
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
    sess.log_params({"evaluator": evaluator.to_dict()})
