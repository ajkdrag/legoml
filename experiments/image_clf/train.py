from dataclasses import asdict
from pathlib import Path

import torch

from experiments.data_utils import create_dataloaders
from experiments.image_clf.config import Config
from experiments.image_clf.models import (
    ConvMixer_w256_d8_p2_k5,
    ConvNeXt_tiny_32x32,
    MobileNet_tiny_32x32,
    Res2NetWide_32x32,
    ResNetWide_tiny_32x32,
)
from experiments.image_clf.steps import eval_step, train_step
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
    max_epochs=25,
    train_bs=256,
    eval_bs=128,
    train_log_interval=50,
)
train_dl, eval_dl = create_dataloaders("cifar10", config, "classification")
model = Res2NetWide_32x32()
summary = summarize_model(model, next(iter(train_dl)).inputs, depth=2)

# eval stuff
model.to(device)
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


# optim and scheduler stuff
init_lr = 0.01
max_lr = 0.3
total_steps = config.max_epochs * len(train_dl)

param_groups = default_groups(model, lr=init_lr, weight_decay=5e-4)
print_param_groups(model, param_groups)

optimizer = torch.optim.SGD(
    param_groups,
    momentum=0.9,
    nesterov=True,
)


def sch_best_fn():
    return evaluator.state.metrics["eval_acc"]


warmup_scheduler = KeyframeLR(
    optimizer=optimizer,
    frames=[
        Keyframe(0, init_lr),
        LinearInterpolation(),
        Keyframe((0.1 * total_steps) - 1, max_lr),
        CosineInterpolation(),
        Keyframe("end", 1e-3),
    ],
    end=(config.max_epochs * len(train_dl)) - 1,
    units="steps",
)

print(warmup_scheduler.schedules)


def run_eval(*, context: Context, state: EngineState):
    evaluator.loop()
    logger.info("Evaluation complete", epoch=state.epoch)


def sched_step(*, context: Context, state: EngineState, **kwargs):
    if state.global_step <= warmup_scheduler.end:
        warmup_scheduler.step()


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

    trainer.add_event_handler(
        event=Events.EPOCH_END,
        handler=run_eval,
    )

    trainer.add_event_handler(
        Events.STEP_END,
        handler=sched_step,
    )

    trainer.callbacks += [
        CheckpointCallback(
            dirpath=sess.get_artifact_dir().joinpath("checkpoints"),
            save_every_n_epochs=config.max_epochs + 5,  # don't save every epoch
            save_on_engine_end=True,
            best_fn=lambda: evaluator.state.metrics["eval_acc"],
        ),
    ]

    trainer.loop()
    sess.log_params({"exp_config": asdict(config)})
    sess.log_text("model", f"{summary}\n\n{model}")
    sess.log_params({"trainer": trainer.to_dict()})
    sess.log_params({"evaluator": evaluator.to_dict()})
