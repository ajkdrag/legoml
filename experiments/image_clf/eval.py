from dataclasses import asdict
from pathlib import Path

import torch

from experiments.image_clf.config import Config
from experiments.data_utils import create_dataloaders
from experiments.image_clf.models import ResNetBasic_tiny_32x32
from experiments.image_clf.steps import eval_step
from legoml.callbacks.metric import MetricsCallback
from legoml.core.context import Context
from legoml.core.engine import Engine
from legoml.metrics.multiclass import MultiClassAccuracy
from legoml.utils.log import get_logger
from legoml.utils.seed import set_seed
from legoml.utils.track import run

logger = get_logger(__name__)
device = torch.device("mps")
set_seed(42)
config = Config(max_epochs=1)


model = ResNetBasic_tiny_32x32().to(device)
train_dl, eval_dl = create_dataloaders("cifar10", config, "classification")


with run(base_dir=Path("runs").joinpath("eval_img_clf_cifar10")) as sess:
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

    checkpoint_path = "./runs/train_img_clf_cifar10/run_20250829_220515/artifacts/checkpoints/ckpt_best.pt"

    evaluator.load_checkpoint(checkpoint_path=checkpoint_path)
    evaluator.state.reset()
    evaluator.loop(train_dl, config.max_epochs)

    sess.log_params({"exp_config": asdict(config)})
    sess.log_text("model", str(model))
    sess.log_params({"evaluator": evaluator.to_dict()})
