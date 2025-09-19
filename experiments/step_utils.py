import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from legoml.core.engine import Engine
from legoml.utils.log import get_logger

logger = get_logger(__name__)


def forward_and_compute_loss(model, loss_fn, inputs, targets, device, use_amp=False):
    """Forward pass and loss computation with optional AMP."""
    inputs, targets = inputs.to(device), targets.to(device)

    with torch.autocast(enabled=use_amp, device_type=device.type, dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

    return outputs, loss


def backward_and_step(
    loss, model: nn.Module, optimizer: Optimizer, scaler=None, clip_norm=True
):
    """Backward pass and optimizer step with optional AMP."""
    if scaler is not None:
        scaler.scale(loss).backward()
        if clip_norm:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()


def log_step(engine: Engine, mode, log_interval: int):
    """Log loss if at the right interval."""
    if engine.state.local_step % log_interval == 0:
        loss = engine.state.output.loss_scalar
        lr = engine.context.get_lr()
        logger.info(
            f"Loss: {loss}",
            step=engine.state.local_step,
            mode=mode,
            epoch=engine.state.epoch,
            lr=lr,
        )
