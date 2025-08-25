from pathlib import Path

import torch


def load_checkpoint(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def load_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
):
    ckpt = load_checkpoint(checkpoint_path, device=device)
    model_sd = ckpt.get("model")
    if model_sd:
        model.load_state_dict(model_sd)
    return ckpt


def save_checkpoint(state_dict: dict, path: str | Path):
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state_dict, path)
    return path
