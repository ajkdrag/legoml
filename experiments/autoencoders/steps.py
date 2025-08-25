import torch
from legoml.core.engine import Engine
from legoml.core.context import Context
from legoml.core.step_output import StepOutput
from legoml.utils.logging import get_logger
from legoml.data.batches import AutoencoderBatch
from experiments.image_clf.config import Config

logger = get_logger(__name__)


def train_step(engine: Engine, batch: AutoencoderBatch, context: Context) -> StepOutput:
    model = context.model
    loss_fn = context.loss_fn
    optimizer = context.optimizer
    use_amp = context.scaler is not None
    device = context.device
    config: Config = context.config

    assert optimizer is not None, "Optimizer is not set"

    inputs, targets = batch.inputs.to(device), batch.inputs.to(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(enabled=use_amp, device_type=device.type):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

    if context.scaler is not None:
        context.scaler.scale(loss).backward()
        context.scaler.step(optimizer)
        context.scaler.update()
    else:
        loss.backward()
        optimizer.step()

    if engine.state.local_step % config.train_log_interval == 0:
        logger.info(
            f"Loss: {loss.item()}",
            step=engine.state.local_step,
            mode="train",
        )
    return StepOutput(loss=loss)


def eval_step(engine: Engine, batch: AutoencoderBatch, context: Context) -> StepOutput:
    model = context.model
    loss_fn = context.loss_fn
    device = context.device
    config: Config = context.config

    inputs, targets = batch.inputs.to(device), batch.inputs.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

    if engine.state.local_step % config.eval_log_interval == 0:
        logger.info(
            f"Loss: {loss.item()}",
            step=engine.state.local_step,
            mode="eval",
        )

    return StepOutput(loss=loss)
