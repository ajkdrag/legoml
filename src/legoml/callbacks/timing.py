import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import csv

from legoml.core.context import Context
from legoml.core.callback import Callback, implements
from legoml.core.state import EngineState
from legoml.utils.logging import get_logger

logger = get_logger(__name__)


@implements(
    "on_engine_start", "on_step_start", "on_step_end", "on_epoch_end", "on_engine_end"
)
class TimingCallback(Callback):
    def __init__(
        self,
        log_interval: int = 10,
        save_detailed_stats: bool = True,
        output_dir: str | Path = "./timing_logs",
    ):
        """
        Callback for measuring step execution times and identifying performance bottlenecks.

        Args:
            log_interval: How often to log timing statistics (in steps)
            save_detailed_stats: Whether to save detailed timing data to CSV
            output_dir: Directory to save timing logs
        """
        self.log_interval = log_interval
        self.save_detailed_stats = save_detailed_stats
        self.output_dir = Path(output_dir)
        if save_detailed_stats:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Timing data
        self.step_start_time: float = 0.0
        self.step_times: List[float] = []
        self.epoch_times: List[float] = []
        self.detailed_timings: List[Dict] = []

        # Statistics tracking
        self.current_epoch_times: List[float] = []
        self.slowest_steps: List[tuple] = []  # (epoch, step, time)
        self.epoch_start_time: float = 0.0

    def on_engine_start(self, context: Context, state: EngineState):
        self.step_times = []
        self.epoch_times = []
        self.detailed_timings = []
        logger.info("Timing callback started - tracking step execution times")

    def on_step_start(self, context: Context, state: EngineState, batch):
        self.step_start_time = time.perf_counter()

    def on_step_end(self, context: Context, state: EngineState, batch):
        step_time = time.perf_counter() - self.step_start_time
        self.step_times.append(step_time)
        self.current_epoch_times.append(step_time)

        # Track slowest steps for analysis
        self.slowest_steps.append((state.epoch, state.local_step, step_time))
        self.slowest_steps.sort(key=lambda x: x[2], reverse=True)
        if len(self.slowest_steps) > 10:  # Keep only top 10 slowest
            self.slowest_steps = self.slowest_steps[:10]

        # Save detailed timing data
        if self.save_detailed_stats:
            timing_record = {
                "epoch": state.epoch,
                "global_step": state.global_step,
                "local_step": state.local_step,
                "step_time": step_time,
                "loss": state.output.loss_scalar or "unknown",
                "batch_size": len(batch[0])
                if isinstance(batch, (list, tuple)) and len(batch) > 0
                else "unknown",
            }

            self.detailed_timings.append(timing_record)

        # Log periodic statistics
        if (state.local_step + 1) % self.log_interval == 0:
            recent_times = self.current_epoch_times[-self.log_interval :]
            avg_time = sum(recent_times) / len(recent_times)
            min_time = min(recent_times)
            max_time = max(recent_times)

            # Calculate samples per second (assuming batch dimension is first)
            samples_per_sec = "unknown"
            if isinstance(batch, (list, tuple)) and len(batch) > 0:
                batch_size = len(batch[0])
                samples_per_sec = f"{batch_size / avg_time:.1f}"

            logger.info(
                f"Step timing - Avg: {avg_time * 1000:.1f}ms, "
                f"Min: {min_time * 1000:.1f}ms, Max: {max_time * 1000:.1f}ms, "
                f"Samples/sec: {samples_per_sec}",
                epoch=state.epoch,
                step=state.local_step,
            )

    def on_epoch_end(self, context: Context, state: EngineState):
        if not self.current_epoch_times:
            return

        epoch_total_time = sum(self.current_epoch_times)
        epoch_avg_step = sum(self.current_epoch_times) / len(self.current_epoch_times)
        epoch_min_step = min(self.current_epoch_times)
        epoch_max_step = max(self.current_epoch_times)

        self.epoch_times.append(epoch_total_time)

        # Calculate standard deviation
        mean = epoch_avg_step
        variance = sum((t - mean) ** 2 for t in self.current_epoch_times) / len(
            self.current_epoch_times
        )
        std_dev = variance**0.5

        logger.info(
            f"Epoch {state.epoch} timing summary - "
            f"Total: {epoch_total_time:.2f}s, "
            f"Avg step: {epoch_avg_step * 1000:.1f}ms, "
            f"Min step: {epoch_min_step * 1000:.1f}ms, "
            f"Max step: {epoch_max_step * 1000:.1f}ms, "
            f"Std dev: {std_dev * 1000:.1f}ms"
        )

        # Reset for next epoch
        self.current_epoch_times = []

    def on_engine_end(self, context: Context, state: EngineState):
        if not self.step_times:
            logger.warning("No timing data collected")
            return

        # Calculate overall statistics
        total_time = sum(self.step_times)
        avg_step_time = total_time / len(self.step_times)
        min_step_time = min(self.step_times)
        max_step_time = max(self.step_times)

        logger.info(
            f"Training timing summary - "
            f"Total steps: {len(self.step_times)}, "
            f"Total time: {total_time:.2f}s, "
            f"Avg step: {avg_step_time * 1000:.1f}ms, "
            f"Min step: {min_step_time * 1000:.1f}ms, "
            f"Max step: {max_step_time * 1000:.1f}ms"
        )

        # Log slowest steps for investigation
        if self.slowest_steps:
            logger.info("Top 5 slowest steps:")
            for i, (epoch, step, time_taken) in enumerate(self.slowest_steps[:5]):
                logger.info(
                    f"  {i + 1}. Epoch {epoch}, Step {step}: {time_taken * 1000:.1f}ms"
                )

        # Save detailed timing data to CSV
        if self.save_detailed_stats and self.detailed_timings:
            csv_file = self.output_dir / f"detailed_timings_epoch_{state.epoch}.csv"

            fieldnames = [
                "epoch",
                "global_step",
                "local_step",
                "step_time",
                "batch_size",
            ]
            if any("loss" in record for record in self.detailed_timings):
                fieldnames.append("loss")

            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.detailed_timings)

            logger.info(f"Detailed timing data saved to {csv_file}")

    def get_statistics(self) -> Dict:
        """Get current timing statistics as a dictionary."""
        if not self.step_times:
            return {}

        return {
            "total_steps": len(self.step_times),
            "total_time_sec": sum(self.step_times),
            "avg_step_time_ms": (sum(self.step_times) / len(self.step_times)) * 1000,
            "min_step_time_ms": min(self.step_times) * 1000,
            "max_step_time_ms": max(self.step_times) * 1000,
            "slowest_steps": [
                (epoch, step, time_ms * 1000)
                for epoch, step, time_ms in self.slowest_steps[:5]
            ],
        }

