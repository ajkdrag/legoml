from legoml.core.tasks.train_img_clf import TrainerForImageClassificationNode
from legoml.core.tasks.eval_img_clf import EvaluatorForImageClassificationNode
from legoml.core.models.tinycnn import TinyCNNNode
from legoml.core.datasets.mnist_clf import MNISTNode
from legoml.core.optimizers import AdamNode
from legoml.core.schedulers import CosineAnnealingNode
from legoml.core.blocks.convs import ConvBlockNode
from legoml.utils.logging import get_logger, bind
from legoml.core.dataloaders import DataLoaderForClassificationNode


logger = get_logger(__name__)


def train_tiny_cnn_baseline():
    """Baseline TinyCNN configuration."""
    exp_name = "tiny_cnn_baseline"
    bind(experiment_name=exp_name)
    logger.info("Starting TinyCNN baseline training experiment")

    node = TrainerForImageClassificationNode(
        name="tiny_cnn_baseline",
        input_shape=(1, 28, 28),
        model=TinyCNNNode(
            num_classes=10,
            conv_blocks=[
                ConvBlockNode(in_channels=1, out_channels=32, pooling=None),
                ConvBlockNode(in_channels=32, out_channels=64),
                ConvBlockNode(in_channels=64, out_channels=128),
            ],
        ),
        dataset=MNISTNode(augmentation=False, num_classes=10),
        train_dl=DataLoaderForClassificationNode(shuffle=True, batch_size=64),
        val_dl=DataLoaderForClassificationNode(shuffle=False),
        optimizer=AdamNode(lr=1e-3, weight_decay=1e-4),
        scheduler=CosineAnnealingNode(T_max=10),
        epochs=1,
        save_frequency=1,
        gradient_clip_val=1.0,
        device="mps",
    )

    logger.info(
        "Experiment node created: ",
        config=node.to_dict(),
    )

    trainer = node.build()
    logger.info("Experiment configuration built")

    results = trainer.train()
    logger.info("Training completed successfully", results=results)

    return results


def eval_tiny_cnn_baseline():
    """Baseline TinyCNN evaluation."""
    exp_name = "tiny_cnn_baseline_eval"
    bind(experiment_name=exp_name)
    logger.info("Starting TinyCNN baseline eval experiment")

    node = EvaluatorForImageClassificationNode(
        name="tiny_cnn_baseline",
        model=TinyCNNNode(
            num_classes=10,
            conv_blocks=[
                ConvBlockNode(in_channels=1, out_channels=32, pooling=None),
                ConvBlockNode(in_channels=32, out_channels=64),
                ConvBlockNode(in_channels=64, out_channels=128),
            ],
        ),
        checkpoint="./models/tiny_cnn_baseline_epoch_1.pt",
        dataset=MNISTNode(augmentation=False, num_classes=10),
        test_dl=DataLoaderForClassificationNode(shuffle=False, batch_size=64),
        split="val",
        device="mps",
        detailed_metrics=True,
    )

    logger.info(
        "Experiment node created: ",
        config=node.to_dict(),
    )

    evaluator = node.build()
    logger.info("Experiment configuration built")

    results = evaluator.evaluate()
    logger.info("Evaluation completed successfully", results=results)

    return results
