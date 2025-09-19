from dataclasses import asdict
from functools import partial
from math import inf
from pathlib import Path

import torch

from experiments.data_utils import create_dataloaders
from experiments.image_clf.config import Config
from experiments.image_clf.models import (
    CNN__MLP_tiny_32x32,
    ConvMixer_w256_d8_p2_k5,
    ConvNeXt_2x2_stem,
    ConvNeXt_tiny_32x32,
    MobileNet_tiny_32x32,
    Res2Net_32x32,
    Res2NetWide_32x32,
    ResNetWide_tiny_32x32,
)
from experiments.image_clf.steps import eval_step, train_step, update_bn
from legoml.callbacks.checkpoint import CheckpointCallback
from legoml.callbacks.metric import MetricsCallback
from legoml.core.context import Context
from legoml.core.engine import Engine
from legoml.core.event import Events
from legoml.core.state import EngineState
from legoml.metrics.multiclass import MultiClassAccuracy
from legoml.schedulers.keyframe import (
    CosineInterpolation,
    Keyframe,
    KeyframeLR,
    LinearInterpolation,
)
from legoml.utils.device import get_device
from legoml.utils.ema import ModelEma
from legoml.utils.log import get_logger
from legoml.utils.optim import default_groups, print_param_groups
from legoml.utils.seed import set_seed
from legoml.utils.summary import summarize_model
from legoml.utils.track import run

logger = get_logger(__name__)

set_seed(42)
device = get_device()
logger.info("Using device: %s", device.type)

config = Config(
    train_augmentation=True,
    max_epochs=100,
    train_bs=256,
    eval_bs=128,
    train_log_interval=50,
    eval_interval=2,
    eval_ema_interval=5,
)
train_dl, eval_dl = create_dataloaders("cifar10", config, "classification")
model = ConvMixer_w256_d8_p2_k5()
summary = summarize_model(model, next(iter(train_dl)).inputs, depth=2)

# eval stuff
model.to(device)
ema = ModelEma(model, decay=0.999)
evaluator = Engine(
    eval_step,
    Context(
        config=config,
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=device,
    ),
    state=EngineState(max_epochs=1, dataloader=eval_dl),
    callbacks=[
        MetricsCallback(metrics=[MultiClassAccuracy("eval_acc")]),
    ],
)

ema_evaluator = Engine(
    eval_step,
    Context(
        config=config,
        model=ema._model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=device,
    ),
    state=EngineState(max_epochs=1, dataloader=eval_dl),
    callbacks=[
        MetricsCallback(metrics=[MultiClassAccuracy("ema_eval_acc")]),
    ],
)


# optim and scheduler stuff
init_lr = 0.01
max_lr = 0.3
total_steps = config.max_epochs * len(train_dl)

param_groups = default_groups(model, lr=init_lr, weight_decay=0.001)
print_param_groups(model, param_groups)

optimizer = torch.optim.SGD(
    param_groups,
    momentum=0.9,
    nesterov=True,
)


def sch_best_fn():
    return evaluator.state.metrics["eval_acc"]


scheduler = KeyframeLR(
    optimizer=optimizer,
    frames=[
        Keyframe(0, init_lr),
        LinearInterpolation(),
        Keyframe((0.1 * total_steps) - 1, max_lr),
        CosineInterpolation(),
        Keyframe("end", 1e-3),
    ],
    end=total_steps - 1,
    units="steps",
)

logger.info(scheduler.schedules)


def run_eval(*, context: Context, state: EngineState):
    if state.epoch % config.eval_interval == 0:
        evaluator.loop()
        logger.info("Evaluation complete", epoch=state.epoch)


def run_ema_eval(*, context: Context, state: EngineState, dirpath: str | Path):
    if (
        state.epoch > int(config.max_epochs * 0.25)
        and state.epoch % config.eval_ema_interval == 0
    ):
        update_bn(
            model=ema._model,
            dl=train_dl,
            device=device,
            max_batches=len(train_dl),
        )
        ema_evaluator.loop()
        logger.info("EMA Evaluation complete", epoch=state.epoch)

        # Saving if better than eval metric
        ema_eval_metric = ema_evaluator.state.metrics["ema_eval_acc"]
        eval_metric = evaluator.state.metrics.get("eval_acc", float(-inf))
        if ema_eval_metric > eval_metric:
            logger.info(f"{ema_eval_metric=} is better. Checkpointing...")
            CheckpointCallback._save(
                context=ema_evaluator.context,
                state=ema_evaluator.state,
                path=Path(dirpath) / "ema_ckpt_best.pt",
            )


def ema_copy(*, context: Context, state: EngineState, **kwargs):
    if state.epoch == int(config.max_epochs * 0.25):
        logger.warning("Copying model for EMA...")
        ema.copy(context.model)
        ema_evaluator.context.model = ema._model


def ema_update(*, context: Context, state: EngineState, **kwargs):
    if state.epoch > int(config.max_epochs * 0.25):
        ema.update(model=context.model)


def sched_step(*, context: Context, state: EngineState, **kwargs):
    if state.global_step <= scheduler.end:
        scheduler.step()


with run(base_dir=Path("runs").joinpath("train_img_clf_cifar10")) as sess:
    trainer = Engine(
        train_step,
        context=Context(
            config=config,
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
            device=device,
            scaler=torch.GradScaler(device=device.type),  # slow on M1 air
        ),
        state=EngineState(max_epochs=config.max_epochs, dataloader=train_dl),
        callbacks=[
            MetricsCallback(metrics=[MultiClassAccuracy("train_acc")]),
        ],
    )

    trainer.add_event_handlers(
        Events.STEP_END,
        handlers=[
            ema_update,
            sched_step,
        ],
    )

    trainer.add_event_handlers(
        event=Events.EPOCH_END,
        handlers=[
            run_eval,
            ema_copy,
            partial(
                run_ema_eval, dirpath=sess.get_artifact_dir().joinpath("checkpoints")
            ),
        ],
    )

    trainer.callbacks += [
        CheckpointCallback(
            dirpath=sess.get_artifact_dir().joinpath("checkpoints"),
            save_every_n_epochs=config.max_epochs + 5,  # don't save every epoch
            save_on_engine_end=False,
            save_best=True,
            value_fn=lambda: evaluator.state.metrics.get("eval_acc", float(-inf)),
        ),
    ]

    trainer.loop()
    sess.log_params({"exp_config": asdict(config)})
    sess.log_text("model", f"{summary}\n\n{model}")
    sess.log_params({"trainer": trainer.to_dict()})
    sess.log_params({"evaluator": evaluator.to_dict()})
