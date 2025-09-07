import torch

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


def apply_decoupled_wd(optimizer):
    for group in optimizer.param_groups:
        wd = group.get("weight_decay", 0.0)
        if wd != 0.0:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.add_(p.data, alpha=-wd * group["lr"])


def backward_and_step(loss, optimizer, scheduler=None, scaler=None):
    """Backward pass and optimizer step with optional AMP."""
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    if scheduler:
        scheduler.step()


def log_step(engine: Engine, mode: str, log_interval: int):
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
