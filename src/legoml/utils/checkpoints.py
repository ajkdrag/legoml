import torch
from pathlib import Path
from typing import Dict, Any
from legoml.utils.logging import get_logger


logger = get_logger(__name__)


def load_checkpoint(checkpoint_path: Path | str, device: str = "cpu") -> Dict[str, Any]:
    """Load a model checkpoint from disk."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint", checkpoint_path=str(checkpoint_path))

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        logger.info(
            "Checkpoint loaded successfully",
            epoch=checkpoint.get("epoch", "unknown"),
            metrics=checkpoint.get("metrics", {}),
        )
        return checkpoint
    except Exception as e:
        logger.error("Failed to load checkpoint", error=str(e))
        raise


def load_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path | str,
    device: str = "cpu",
    strict: bool = True,
) -> torch.nn.Module:
    """Load model weights from checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path, device)

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'")

    logger.info("Loading model state from checkpoint")

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        model.to(device)
        logger.info("Model state loaded successfully")
        return model
    except Exception as e:
        logger.error("Failed to load model state", error=str(e))
        raise

