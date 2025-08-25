from dataclasses import asdict
from pathlib import Path

import torch

from experiments.image_clf.config import Config
from experiments.image_clf.data import get_dls
from experiments.image_clf.models import CNN__MLP_tiny_28x28
from experiments.image_clf.steps import eval_step
from legoml.callbacks.metric import MetricsCallback
from legoml.core.context import Context
from legoml.core.engine import Engine
from legoml.metrics.multiclass import MultiClassAccuracy
from legoml.utils.logging import get_logger
from legoml.utils.seed import set_seed
from legoml.utils.track import run

logger = get_logger(__name__)
device = torch.device("mps")
set_seed(42)
config = Config(max_epochs=1)


model = CNN__MLP_tiny_28x28().to(device)
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

    checkpoint_path = (
        "./runs/train_img_clf/run_20250824_183653/"
        + "artifacts/checkpoints/ckpt_last.pt"
    )

    evaluator.load_checkpoint(checkpoint_path=checkpoint_path)
    evaluator.state.reset()
    evaluator.loop(eval_dl, config.max_epochs)

    sess.log_params({"exp_config": asdict(config)})
    sess.log_text("model", str(model))
    sess.log_params({"evaluator": evaluator.to_dict()})
