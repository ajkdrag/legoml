from dataclasses import dataclass
import torch
from legoml.utils.logging import get_logger
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader, Subset
from legoml.core.engine import Engine
from legoml.core.context import Context
from legoml.metrics.multiclass import MultiClassAccuracy
from legoml.nn.composites.tinycnn import TinyCNNNode
from legoml.nn.mlp import MLPNode
from legoml.nn.activations import ReluNode
from legoml.nn.base import NoopNode
from legoml.data.mnist import MNISTConfig, build_mnist
from legoml.utils.seed import set_seed
from legoml.callbacks.eval import EvalOnEpochEndCallback
from legoml.callbacks.metric import MetricsCallback
from legoml.core.step_output import StepOutput


logger = get_logger(__name__)
device = torch.device("mps")
logger.info(f"Using device: {device.type}")
set_seed(42)


@dataclass
class Config:
    batch_size: int = 64
    train_augmentation: bool = False
    max_epochs: int = 2
    train_log_interval: int = 100  # steps


# MODEL
node = TinyCNNNode(
    input_channels=1,
    mlp=MLPNode(
        dims=[128, 10],
        activation=ReluNode(),
        last_activation=NoopNode(),
    ),
)
model = node.build().to(device)

# DATA
train_ds = build_mnist(
    MNISTConfig(
        split="train",
        augmentation=Config.train_augmentation,
    )
)


val_ds = build_mnist(
    MNISTConfig(
        split="test",
        augmentation=False,
    )
)

train_loader = DataLoader(
    # one batch
    train_ds,
    batch_size=Config.batch_size,
    shuffle=True,
)
eval_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
)

logger.info(
    "Created data loaders",
    train_size=len(train_loader),
    val_size=len(eval_loader),
)

# TRAINING
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)
scheduler = lrs.CosineAnnealingLR(
    optimizer,
    T_max=Config.max_epochs,
    eta_min=1e-6,
)


def train_step(engine: Engine, batch, context: Context) -> StepOutput:
    model = context.model
    loss_fn = context.loss_fn
    optimizer = context.optimizer
    use_amp = context.scaler is not None
    device = context.device

    assert optimizer is not None, "Optimizer is not set"

    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)

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

    if engine.state.local_step % Config.train_log_interval == 0:
        logger.info(
            f"Loss: {loss.item()}",
            step=engine.state.local_step,
            mode="train",
        )
    return StepOutput(
        loss=loss,
        predictions=outputs.detach().cpu(),
        targets=targets.detach().cpu(),
        metadata={"loss_scalar": loss.item()},
    )


def eval_step(engine: Engine, batch, context: Context) -> StepOutput:
    model = context.model
    loss_fn = context.loss_fn
    device = context.device

    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

    return StepOutput(
        loss=loss,
        predictions=outputs.detach().cpu(),
        targets=targets.detach().cpu(),
        metadata={"loss_scalar": loss.item()},
    )


train_context = Context(
    model=model,
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    scaler=torch.GradScaler(device=device.type),
)
trainer = Engine(train_step, train_context)

eval_context = Context(
    model=model,
    loss_fn=torch.nn.CrossEntropyLoss(),
    device=device,
)
evaluator = Engine(eval_step, eval_context)


trainer.callbacks.extend(
    [
        EvalOnEpochEndCallback(evaluator, eval_loader, 1),
        MetricsCallback(metrics=[MultiClassAccuracy()]),
    ]
)

evaluator.callbacks.extend(
    [
        MetricsCallback(metrics=[MultiClassAccuracy()]),
    ]
)

trainer.loop(train_loader, max_epochs=Config.max_epochs)
logger.info(
    "Training complete",
    train_metrics=trainer.state.metrics,
    eval_metrics=evaluator.state.metrics,
)
