import torch
import os
import shutil


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    run_name,
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
    checkpoint_path = os.path.join(save_dir, f"{run_name}_checkpoint.pth")
    best_model_path = os.path.join(save_dir, f"{run_name}_best_model.pth")

    if is_best:
        torch.save(state, best_model_path)
        print(f"Best model saved to {best_model_path}")
    else:
        torch.save(state, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path
