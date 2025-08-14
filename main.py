from legoml.experiments import (
    train_tiny_cnn_baseline,
    eval_tiny_cnn_baseline,
)
from legoml.utils.logging import setup_logging

setup_logging(log_level="INFO", structured=False)


if __name__ == "__main__":
    results = eval_tiny_cnn_baseline()
