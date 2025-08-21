from pathlib import Path
from dataclasses import asdict
import torch

from legoml.callbacks.checkpoint import CheckpointCallback
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

from experiments.image_clf.config import Config
from experiments.image_clf.data import get_dls
from experiments.image_clf.steps import eval_step

logger = get_logger(__name__)
device = torch.device("mps")
set_seed(42)
config = Config(max_epochs=1)

node = TinyCNNNode(
    input_channels=1,
    mlp=MLPNode(
        dims=[128, 10],
        activation=ReluNode(),
        last_activation=NoopNode(),
    ),
)
model = node.build().to(device)
train_dl, eval_dl = get_dls(config)


with run(base_dir=Path("runs").joinpath("eval_img_clf")) as sess:
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

    checkpoint_path = "./checkpoints/ckpt_epoch_1.pt"
    CheckpointCallback.load_into(
        context=eval_context,
        state=evaluator.state,
        path=checkpoint_path,
        map_location=device.type,
    )

    evaluator.state.reset()
    evaluator.loop(eval_dl, config.max_epochs)

    sess.log_params({"exp_config": asdict(config)})
    sess.log_params({"model": asdict(node)})
    sess.log_params({"evaluator": evaluator.to_dict()})
