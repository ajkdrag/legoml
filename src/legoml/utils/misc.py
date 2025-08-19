import torch
import os
from pathlib import Path


def load_checkpoint(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> dict:
    """Load a checkpoint from disk."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    basename,
    save_dir,
    is_best,
    metadata,
):
    """Save a checkpoint for the model."""

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "metadata": metadata,
    }
    checkpoint_path = os.path.join(save_dir, f"{basename}_ep_{epoch}.pth")
    best_model_path = os.path.join(save_dir, f"{basename}_best_ep_{epoch}.pth")

    if is_best:
        torch.save(state, best_model_path)
        print(f"Best model saved to {best_model_path}")
    else:
        torch.save(state, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path
