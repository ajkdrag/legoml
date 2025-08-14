import torchsummary
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from legoml.core.base import (
    DatasetForClassificationNode,
    ModelNode,
    Node,
    OptimizerNode,
    SchedulerNode,
)
from legoml.core.dataloaders import DataLoaderForClassificationNode
from legoml.utils.checkpoints import load_model_from_checkpoint
from legoml.utils.logging import bind, get_logger


@dataclass(kw_only=True)
class TrainerForImageClassificationNode(Node):
    """Node for supervised learning trainers."""

    name: str
    model: ModelNode
    checkpoint: str | None = None
    input_shape: tuple[int, ...]

    dataset: DatasetForClassificationNode
    train_dl: DataLoaderForClassificationNode
    val_dl: DataLoaderForClassificationNode

    optimizer: OptimizerNode
    scheduler: SchedulerNode | None

    epochs: int = 2
    gradient_clip_val: float | None = None
    log_every_n_steps: int = 100
    val_frequency: int = 1
    save_frequency: int = 5
    early_stopping_patience: int | None = None

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./models"

    def build(self):
        return TrainerForImageClassification(self)


@dataclass
class BatchResults:
    loss: float
    metrics: dict
    count_correct: int
    count_samples: int


@dataclass
class EpochResults:
    loss: float
    metrics: dict


class TrainerForImageClassification:
    logger = get_logger("trainer.image_classification")

    def __init__(self, config: TrainerForImageClassificationNode):
        self.config = config

        self.logger.info("Building model")
        self.model = self.config.model.build()

        if self.config.checkpoint:
            self.model = load_model_from_checkpoint(
                self.model,
                self.config.checkpoint,
            )

        torchsummary.summary(self.model, self.config.input_shape)

        self.logger.info("Building data loaders")

        self.train_ds = self.config.dataset.build(split="train")
        self.val_ds = self.config.dataset.build(split="val")
        self.train_loader = self.config.train_dl.build(self.train_ds)
        self.val_loader = self.config.val_dl.build(self.val_ds)

        self.logger.info("Building optimizer")
        self.optimizer = self.config.optimizer.build(self.model.parameters())

        if self.config.scheduler:
            self.logger.info("Building scheduler")
            self.scheduler = self.config.scheduler.build(self.optimizer)
        else:
            self.scheduler = None

        self.train_batches = len(self.train_loader)
        self.val_batches = len(self.val_loader)
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Trainer initialization complete",
            train_batches=self.train_batches,
            val_batches=self.val_batches,
            device=str(self.device),
            save_dir=str(self.save_dir),
        )

    def _calculate_metrics(
        self,
        total_correct: int,
        total_samples: int,
    ) -> dict[str, float]:
        """Calculate common metrics for training/validation."""
        accuracy = 100.0 * total_correct / total_samples
        return {
            "accuracy": accuracy,
            "total_samples": total_samples,
        }

    def _process_batch(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> BatchResults:
        """Process a single batch"""
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        loss = F.cross_entropy(output, target)

        pred = output.argmax(dim=1, keepdim=True)
        batch_correct = pred.eq(target.view_as(pred)).sum().item()
        batch_size = data.size(0)
        batch_metrics = self._calculate_metrics(batch_correct, batch_size)

        return BatchResults(
            loss=loss,
            metrics=batch_metrics,
            count_correct=batch_correct,
            count_samples=batch_size,
        )

    def _checkpoint(
        self, epoch: int, metrics: dict[str, float], is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if is_best:
            save_path = self.save_dir / f"{self.config.name}_best.pt"
        else:
            save_path = self.save_dir / f"{self.config.name}_epoch_{epoch + 1}.pt"
        self.logger.info("Saving model checkpoint", save_path=str(save_path))
        torch.save(checkpoint, save_path)

    def train_epoch(self) -> EpochResults:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        train_pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Training",
            unit="batch",
            leave=False,
        )

        for batch_idx, (data, target) in train_pbar:
            bind(batch_idx=batch_idx, total_batches=self.train_batches)

            self.optimizer.zero_grad()
            batch_results = self._process_batch(data, target)
            batch_results.loss.backward()

            if self.config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val,
                )

            self.optimizer.step()

            total_loss += batch_results.loss.item()
            total_correct += batch_results.count_correct
            total_samples += batch_results.count_samples

            train_pbar.set_postfix(
                {
                    "batch_loss": f"{batch_results.loss.item():.4f}",
                    **batch_results.metrics,
                }
            )

            if batch_idx % self.config.log_every_n_steps == 0:
                self.logger.info(
                    "Training batch completed",
                    batch_loss=batch_results.loss.item(),
                    **batch_results.metrics,
                )

        if self.scheduler is not None:
            old_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]["lr"]
            if old_lr != new_lr:
                self.logger.debug("Learning rate updated", old_lr=old_lr, new_lr=new_lr)

        epoch_metrics = self._calculate_metrics(total_correct, total_samples)
        epoch_results = EpochResults(
            loss=total_loss / total_samples,
            metrics=epoch_metrics,
        )

        self.logger.info(
            "Training epoch completed",
            epoch_loss=epoch_results.loss,
            **epoch_results.metrics,
        )

        return epoch_results

    def validate(self) -> EpochResults:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        val_pbar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc="Validation",
            unit="batch",
            leave=False,
        )

        with torch.no_grad():
            for batch_idx, (data, target) in val_pbar:
                bind(batch_idx=batch_idx, total_batches=self.val_batches)

                batch_results = self._process_batch(data, target)

                total_loss += batch_results.loss.item()
                total_correct += batch_results.count_correct
                total_samples += batch_results.count_samples

                val_pbar.set_postfix(
                    {
                        "batch_loss": f"{batch_results.loss.item():.4f}",
                        **batch_results.metrics,
                    }
                )

        epoch_metrics = self._calculate_metrics(total_correct, total_samples)
        epoch_results = EpochResults(
            loss=total_loss / total_samples,
            metrics=epoch_metrics,
        )

        self.logger.info(
            "Validation epoch completed",
            epoch_loss=epoch_results.loss,
            **epoch_results.metrics,
        )

        return epoch_results

    def train(self) -> dict[str, Any]:
        """Train the model for all epochs."""
        results = {
            "train_history": [],
            "val_history": [],
            "final_train_metrics": {},
            "final_val_metrics": {},
        }

        self.logger.info("Starting training", epochs=self.config.epochs)

        epoch_pbar = tqdm(
            range(self.config.epochs),
            desc="Training Progress",
            unit="epoch",
        )

        for epoch in epoch_pbar:
            bind(epoch=epoch + 1, total_epochs=self.config.epochs)

            epoch_pbar.set_description(
                f"Epoch {epoch + 1}/{self.config.epochs}",
            )

            self.logger.info(
                "Starting epoch",
                epoch=epoch + 1,
                total_epochs=self.config.epochs,
            )

            train_results = self.train_epoch()
            results["train_history"].append(train_results)
            results["final_train_metrics"] = train_results.metrics

            if (epoch + 1) % self.config.val_frequency == 0:
                val_results = self.validate()
                results["val_history"].append(val_results)
                results["final_val_metrics"] = val_results.metrics

            if (epoch + 1) % self.config.save_frequency == 0:
                self._checkpoint(epoch, train_results.metrics)

            epoch_pbar.set_postfix(
                {
                    "train_loss": f"{train_results.loss:.4f}",
                    **train_results.metrics,
                }
            )

        self.logger.info(
            "Training completed successfully",
            final_train_metrics=results["final_train_metrics"],
            final_val_metrics=results["final_val_metrics"],
        )

        return results
