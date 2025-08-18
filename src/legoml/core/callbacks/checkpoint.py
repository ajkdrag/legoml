from dataclasses import dataclass
from pathlib import Path
from typing_extensions import override
from legoml.core.contracts.callback import CallbackBase
from legoml.core.state import FitState
from legoml.utils.misc import save_checkpoint


@dataclass
class ModelCheckpoint(CallbackBase):
    save_dir: str = "./models"
    run_name: str = "run"
    every_n_epochs: int = 1

    @override
    def on_train_epoch_end(self, state: FitState):
        epoch = state.epoch
        if epoch % self.every_n_epochs != 0:
            return

        model = state.model
        optimizer = state.optimizer
        scheduler = state.scheduler
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            run_name=self.run_name,
            save_dir=self.save_dir,
            is_best=False,
            metadata={
                "train": state.train_epoch,
                "val": state.eval_epoch,
            },
        )
