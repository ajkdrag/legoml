import torch
from legoml.core.engine import Engine
from legoml.core.context import Context
from legoml.core.step_output import StepOutput
from legoml.data.batches import AutoencoderBatch
from experiments.autoencoders.config import Config
from experiments.step_utils import (
    forward_and_compute_loss,
    backward_and_step,
    log_step_loss,
)


def train_step(engine: Engine, batch: AutoencoderBatch, context: Context) -> StepOutput:
    config: Config = context.config
    model = context.model
    loss_fn = context.loss_fn
    optimizer = context.optimizer
    device = context.device
    use_amp = context.scaler is not None

    assert optimizer is not None, "Optimizer is not set"

    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs, targets = batch.inputs, batch.inputs
    _, loss = forward_and_compute_loss(model, loss_fn, inputs, targets, device, use_amp)

    backward_and_step(loss, optimizer, context.scaler)
    log_step_loss(engine, loss, "train", config.train_log_interval)

    return StepOutput(loss=loss)


def eval_step(engine: Engine, batch: AutoencoderBatch, context: Context) -> StepOutput:
    config: Config = context.config
    model = context.model
    loss_fn = context.loss_fn
    device = context.device

    model.eval()
    inputs, targets = batch.inputs, batch.inputs

    with torch.no_grad():
        _, loss = forward_and_compute_loss(model, loss_fn, inputs, targets, device)

    log_step_loss(engine, loss, "eval", config.eval_log_interval)

    return StepOutput(loss=loss)
