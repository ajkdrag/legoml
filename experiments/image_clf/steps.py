import torch

from experiments.image_clf.config import Config
from experiments.step_utils import (
    backward_and_step,
    forward_and_compute_loss,
    log_step,
)
from legoml.core.context import Context
from legoml.core.engine import Engine
from legoml.core.step_output import StepOutput
from legoml.data.batches import ClassificationBatch


def train_step(
    engine: Engine, batch: ClassificationBatch, context: Context
) -> StepOutput:
    config: Config = context.config
    model = context.model
    loss_fn = context.loss_fn
    optimizer = context.optimizer
    scheduler = context.scheduler
    device = context.device
    use_amp = context.scaler is not None

    assert optimizer is not None, "Optimizer is not set"

    model.train()
    optimizer.zero_grad(set_to_none=True)

    inputs, targets = batch.inputs, batch.targets
    outputs, loss = forward_and_compute_loss(
        model, loss_fn, inputs, targets, device, use_amp
    )

    # scheduler step is handled explicity via engine event handles
    backward_and_step(loss, optimizer, scaler=context.scaler)
    log_step(engine, "train", config.train_log_interval)

    return StepOutput(
        loss=loss,
        predictions=outputs.detach().cpu(),
        targets=targets.detach().cpu(),
    )


def eval_step(
    engine: Engine, batch: ClassificationBatch, context: Context
) -> StepOutput:
    config: Config = context.config
    model = context.model
    loss_fn = context.loss_fn
    device = context.device

    model.eval()
    inputs, targets = batch.inputs, batch.targets

    with torch.no_grad():
        outputs, loss = forward_and_compute_loss(
            model, loss_fn, inputs, targets, device
        )

    log_step(engine, "eval", config.eval_log_interval)

    return StepOutput(
        loss=loss,
        predictions=outputs.detach().cpu(),
        targets=targets.detach().cpu(),
    )
