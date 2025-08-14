from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn.functional as F
from tqdm import tqdm

from legoml.core.base import (
    Node,
    ModelNode,
    DatasetForClassificationNode,
)
from legoml.core.dataloaders import DataLoaderForClassificationNode
from legoml.utils.logging import get_logger, bind
from legoml.utils.checkpoints import load_model_from_checkpoint


@dataclass(kw_only=True)
class EvaluatorForImageClassificationNode(Node):
    """Node for test-only evaluation."""

    name: str
    model: ModelNode
    checkpoint: str

    dataset: DatasetForClassificationNode
    test_dl: DataLoaderForClassificationNode
    split: str = "test"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    detailed_metrics: bool = False

    def build(self):
        return EvaluatorForImageClassification(self)


@dataclass
class EvaluationResults:
    """Results from testing evaluation."""

    loss: float
    metrics: Dict[str, float]
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    confusion_matrix: Optional[torch.Tensor] = None


class EvaluatorForImageClassification:
    """Evaluator for test-only evaluation from checkpoints."""

    logger = get_logger("evaluator.image_classification")

    def __init__(self, config: EvaluatorForImageClassificationNode):
        self.config = config
        self.logger.info("Initializing evaluator")

        self._build_model()
        self._build_data_loader()
        self._setup_device()

    def _build_model(self) -> None:
        """Build and load model from checkpoint."""
        self.logger.info("Building model")
        self.model = self.config.model.build()
        self.model = load_model_from_checkpoint(
            self.model,
            self.config.checkpoint,
        )

    def _build_data_loader(self) -> None:
        """Build test data loader."""
        self.logger.info("Building test data loader")
        self.dataset = self.config.dataset.build(split=self.config.split)
        self.test_loader = self.config.test_dl.build(self.dataset)
        self.test_batches = len(self.test_loader)

    def _setup_device(self) -> None:
        """Setup device and move model."""
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        self.logger.info("Device setup complete", device=str(self.device))

    def _calculate_metrics(
        self, total_correct: int, total_samples: int
    ) -> Dict[str, float]:
        """Calculate basic test metrics."""
        accuracy = 100.0 * total_correct / total_samples
        return {
            "accuracy": accuracy,
            "total_samples": total_samples,
        }

    def _calculate_per_class_metrics(
        self,
        all_preds: torch.Tensor,
        all_targets: torch.Tensor,
        num_classes: int,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate per-class precision, recall, and F1-score."""
        per_class_metrics = {}

        for class_idx in range(num_classes):
            class_mask = all_targets == class_idx
            class_preds = all_preds[class_mask]

            if class_mask.sum() == 0:  # No samples for this class
                continue

            tp = (class_preds == class_idx).sum().item()
            fp = (all_preds == class_idx).sum().item() - tp
            fn = class_mask.sum().item() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            per_class_metrics[f"class_{class_idx}"] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": class_mask.sum().item(),
            }

        return per_class_metrics

    def _calculate_confusion_matrix(
        self,
        all_preds: torch.Tensor,
        all_targets: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Calculate confusion matrix."""
        confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        for pred, target in zip(all_preds, all_targets):
            confusion_matrix[target, pred] += 1
        return confusion_matrix

    def _process_batch(self, data: torch.Tensor, target: torch.Tensor) -> tuple:
        """Process a single batch and return predictions."""
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        loss = F.cross_entropy(output, target)

        pred = output.argmax(dim=1)
        batch_correct = pred.eq(target).sum().item()
        batch_size = data.size(0)

        return loss.item(), batch_correct, batch_size, pred.cpu(), target.cpu()

    def evaluate(self) -> EvaluationResults:
        """Run test evaluation."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        all_predictions = []
        all_targets = []

        test_pbar = tqdm(
            enumerate(self.test_loader),
            total=self.test_batches,
            desc="Testing",
            unit="batch",
        )

        self.logger.info("Starting test evaluation")

        with torch.no_grad():
            for batch_idx, (data, target) in test_pbar:
                bind(batch_idx=batch_idx, total_batches=self.test_batches)

                batch_loss, batch_correct, batch_size, batch_preds, batch_targets = (
                    self._process_batch(data, target)
                )

                total_loss += batch_loss
                total_correct += batch_correct
                total_samples += batch_size

                if self.config.detailed_metrics:
                    all_predictions.append(batch_preds)
                    all_targets.append(batch_targets)

                current_accuracy = 100.0 * batch_correct / batch_size
                test_pbar.set_postfix(
                    {
                        "batch_loss": f"{batch_loss:.4f}",
                        "batch_acc": f"{current_accuracy:.2f}%",
                    }
                )

        basic_metrics = self._calculate_metrics(total_correct, total_samples)
        avg_loss = total_loss / total_samples

        per_class_metrics = None
        confusion_matrix = None

        if self.config.detailed_metrics and all_predictions:
            all_preds = torch.cat(all_predictions)
            all_targs = torch.cat(all_targets)
            num_classes = self.config.dataset.num_classes

            per_class_metrics = self._calculate_per_class_metrics(
                all_preds, all_targs, num_classes
            )
            confusion_matrix = self._calculate_confusion_matrix(
                all_preds, all_targs, num_classes
            )

        results = EvaluationResults(
            loss=avg_loss,
            metrics=basic_metrics,
            per_class_metrics=per_class_metrics,
            confusion_matrix=confusion_matrix,
        )

        self.logger.info(
            "Test evaluation completed",
            test_loss=avg_loss,
            **basic_metrics,
        )

        if per_class_metrics:
            self.logger.info(
                "Per-class metrics calculated", num_classes=len(per_class_metrics)
            )

        return results
