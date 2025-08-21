import torch
import torch.optim.lr_scheduler as lrs

from dataclasses import asdict
from experiments.image_clf.config import Config
from experiments.image_clf.data import get_dls
from experiments.image_clf.steps import eval_step, train_step
from legoml.callbacks.checkpoint import CheckpointCallback
from legoml.callbacks.eval import EvalOnEpochEndCallback
from legoml.callbacks.metric import MetricsCallback
from legoml.core.context import Context
from legoml.core.engine import Engine
from legoml.metrics.multiclass import MultiClassAccuracy
from legoml.nn.activations import ReluNode
from legoml.nn.base import NoopNode
from legoml.nn.composites.tinycnn import TinyCNNNode
from legoml.nn.mlp import MLPNode
from legoml.utils.logging import get_logger
from legoml.utils.seed import set_seed
from legoml.utils.track import run

logger = get_logger(__name__)
device = torch.device("mps")
set_seed(42)
config = Config(train_augmentation=True)


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


node = TinyCNNNode(
    input_channels=1,
    mlp=MLPNode(
        dims=[128, 10],
        activation=ReluNode(),
        last_activation=NoopNode(),
    ),
)
model = node.build().to(device)
optim, sched = build_optim_and_sched(config, model)
train_dl, eval_dl = get_dls(config)

with run("image_clf_train") as sess:
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
                save_on_engine_end=True,
            ),
        ]
    )

    trainer.loop(train_dl, max_epochs=config.max_epochs)
    sess.log_params({"exp_config": asdict(config)})
    sess.log_params({"model": asdict(node)})
    sess.log_params({"trainer": trainer.to_dict()})
    sess.log_params({"evaluator": evaluator.to_dict()})
