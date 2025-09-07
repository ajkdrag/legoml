from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
from torch.utils.data.dataloader import DataLoader

from experiments.data_utils import create_dataloaders
from experiments.image_clf.config import Config
from experiments.image_clf.models import (
    ConvNeXt_SE_32x32,
    MobileNet_tiny_32x32,
    Res2Net_32x32,
)
from experiments.image_clf.steps import eval_step, train_step
from legoml.callbacks.checkpoint import CheckpointCallback
from legoml.callbacks.eval import EvalOnEpochEndCallback
from legoml.callbacks.metric import MetricsCallback
from legoml.core.context import Context
from legoml.core.engine import Engine
from legoml.metrics.multiclass import MultiClassAccuracy
from legoml.nn.ops import LayerScale
from legoml.utils.log import get_logger
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
config = Config(train_augmentation=True, max_epochs=50, train_bs=128)


def separate_parameters(model):
    parameters_decay = set()
    parameters_no_decay = set()
    modules_weight_decay = (nn.Linear, nn.Conv2d)
    modules_no_weight_decay = (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)

    for m_name, m in model.named_modules():
        for param_name, param in m.named_parameters():
            full_param_name = f"{m_name}.{param_name}" if m_name else param_name
            if isinstance(m, modules_no_weight_decay):
                parameters_no_decay.add(full_param_name)
            elif param_name.endswith("bias"):
                parameters_no_decay.add(full_param_name)
            elif isinstance(m, LayerScale) and param_name.endswith("gamma"):
                parameters_no_decay.add(full_param_name)
            elif isinstance(m, modules_weight_decay):
                parameters_decay.add(full_param_name)

    # sanity check
    assert len(parameters_decay & parameters_no_decay) == 0
    assert len(parameters_decay) + len(parameters_no_decay) == len(
        list(model.parameters())
    )

    return parameters_decay, parameters_no_decay


def build_optim_and_sched(
    config: Config,
    model: nn.Module,
    train_dl: DataLoader,
) -> tuple[torch.optim.Optimizer, lrs.LRScheduler]:
    param_dict = {pn: p for pn, p in model.named_parameters()}
    parameters_decay, parameters_no_decay = separate_parameters(model)

    optim_groups = [
        {
            "params": [param_dict[pn] for pn in parameters_decay],
            "weight_decay": 5e-4,
        },
        {"params": [param_dict[pn] for pn in parameters_no_decay], "weight_decay": 0.0},
    ]

    max_lr = 0.1 * (config.train_bs / 256)  # = 0.05 for bs=128

    print(parameters_no_decay)
    optimizer = torch.optim.SGD(
        optim_groups,
        lr=1e-2,
        momentum=0.9,
        nesterov=True,
    )

    scheduler = lrs.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=config.max_epochs,
        steps_per_epoch=len(train_dl),
        pct_start=0.2,
        anneal_strategy="cos",
        three_phase=False,
        div_factor=10,
        final_div_factor=100,
    )
    return optimizer, scheduler


train_dl, eval_dl = create_dataloaders("cifar10", config, "classification")
model = ConvNeXt_SE_32x32()
optim, sched = build_optim_and_sched(config, model, train_dl)

with run(base_dir=Path("runs").joinpath("train_img_clf_cifar10")) as sess:
    train_context = Context(
        config=config,
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
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

    summary = summarize_model(model, next(iter(train_dl)).inputs, depth=2)
    model.to(device)

    trainer.loop(train_dl, max_epochs=config.max_epochs)
    sess.log_params({"exp_config": asdict(config)})
    sess.log_text("model", f"{summary}\n\n{model}")
    sess.log_params({"trainer": trainer.to_dict()})
    sess.log_params({"evaluator": evaluator.to_dict()})
