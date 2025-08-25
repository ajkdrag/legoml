from pathlib import Path
from typing import Any, Dict

from legoml.core.callback import Callback, implements
from legoml.core.context import Context
from legoml.core.state import EngineState
from legoml.utils.io import save_checkpoint
from legoml.utils.logging import get_logger

logger = get_logger(__name__)


@implements("on_engine_start", "on_epoch_end", "on_engine_end")
class CheckpointCallback(Callback):
    def __init__(
        self,
        dirpath: str | Path = "./checkpoints",
        prefix: str = "ckpt",
        save_every_n_epochs: int = 1,
        save_on_engine_end: bool = True,
    ) -> None:
        self.dirpath = Path(dirpath)
        self.prefix = prefix
        self.save_every_n_epochs = max(1, int(save_every_n_epochs))
        self.save_on_engine_end = save_on_engine_end

    def on_engine_start(self, context: Context, state: EngineState) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)
        logger.info(
            "checkpointing",
            dir=str(self.dirpath),
            every_n_epochs=self.save_every_n_epochs,
        )

    def on_epoch_end(self, context: Context, state: EngineState) -> None:
        if state.epoch % self.save_every_n_epochs != 0:
            return
        path = self.dirpath / f"{self.prefix}_epoch_{state.epoch}.pt"
        self._save(context, state, path)

    def on_engine_end(self, context: Context, state: EngineState) -> None:
        if not self.save_on_engine_end:
            return
        path = self.dirpath / f"{self.prefix}_last.pt"
        self._save(context, state, path)

    def _save(self, context: Context, state: EngineState, path: Path) -> None:
        state_dict: Dict[str, Any] = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "local_step": state.local_step,
            "max_epochs": state.max_epochs,
            "metrics": dict(state.metrics),
            "loss": state.output.loss_scalar or "N/A",
            "model": context.model.state_dict(),
            "optimizer": context.optimizer.state_dict() if context.optimizer else None,
            "scheduler": context.scheduler.state_dict() if context.scheduler else None,
            "scaler": context.scaler.state_dict() if context.scaler else None,
        }
        save_checkpoint(state_dict, path)
        logger.info("saved checkpoint", path=str(path), epoch=state.epoch)
