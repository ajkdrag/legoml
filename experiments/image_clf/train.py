from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LRScheduler

from experiments.data_utils import create_dataloaders
from experiments.image_clf.config import Config
from experiments.image_clf.models import (
    CNN__MLP_tiny_32x32,
    ConvNeXt_tiny_32x32,
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
    CosineDecay,
    Keyframe,
    KeyframeLR,
    LinearDecay,
    ReduceLROnMetricPlateau,
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

config = Config(train_augmentation=True, max_epochs=30, train_bs=128)
train_dl, eval_dl = create_dataloaders("cifar10", config, "classification")
model = CNN__MLP_tiny_32x32()
summary = summarize_model(model, next(iter(train_dl)).inputs, depth=2)

# optim stuff
model.to(device)
base_max_lr = 0.1 * (config.train_bs / 128)
param_groups = default_groups(model, lr=base_max_lr, weight_decay=5e-4)
print_param_groups(model, param_groups)

optimizer = torch.optim.SGD(
    param_groups,
    momentum=0.9,
    nesterov=True,
)


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


def sch_best_fn():
    return evaluator.state.metrics["eval_acc"]


scheduler = KeyframeLR(
    optimizer=optimizer,
    frames=[
        Keyframe(0, 0.05),
        CosineDecay(),
        Keyframe(3, 0.1),
        ReduceLROnMetricPlateau(
            best_fn=sch_best_fn,
            patience=3,
            factor=0.5,
            min_lr=1e-2,
            threshold=1e-4,
            threshold_mode="rel",
        ),
        Keyframe("end", 1e-2),
    ],
    end=config.max_epochs,
    units="steps",
)

# scheduler = KeyframeLR(
#     optimizer=optimizer,
#     frames=[
#         Keyframe(0, 0.05),
#         LinearDecay(),
#         Keyframe(3, 1.0),
#         CosineDecay(),
#         Keyframe("end", 1e-4),
#     ],
#     end=config.max_epochs,
#     units="steps",
# )


def run_eval(*, context: Context, state: EngineState, evaluator: Engine):
    evaluator.loop()
    logger.info("Evaluation complete", epoch=state.epoch)


def scheduler_step(*, context: Context, state: EngineState, scheduler: LRScheduler):
    scheduler.step()
    logger.info("Stepped Scheduler", epoch=state.epoch)


with run(base_dir=Path("runs").joinpath("train_img_clf_cifar10")) as sess:
    trainer = Engine(
        train_step,
        context=Context(
            config=config,
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.05),
            device=device,
            scaler=torch.GradScaler(device=device.type),  # slow on M1 air
            scheduler=scheduler,
        ),
        state=EngineState(max_epochs=config.max_epochs, dataloader=train_dl),
        callbacks=[
            MetricsCallback(metrics=[MultiClassAccuracy("train_acc")]),
        ],
    )

    trainer.add_event_handler(
        event=Events.EPOCH_END,
        handler=partial(run_eval, evaluator=evaluator),
    )
    trainer.add_event_handler(
        Events.EPOCH_END,
        handler=partial(scheduler_step, scheduler=scheduler),
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
