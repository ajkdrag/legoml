from legoml.experiments import (
    train_tiny_cnn_v2,
    train_tiny_cnn_v2_with_meta_hook,
)
from legoml.utils.logging import setup_logging

setup_logging(log_level="INFO", structured=False)


if __name__ == "__main__":
    results = train_tiny_cnn_v2_with_meta_hook()
    print(results.train_epoch, results.eval_epoch)
    # results = train_tiny_cnn_baseline()
