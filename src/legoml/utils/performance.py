import time
import os
import psutil
from contextlib import contextmanager
from typing import Dict, Any
import torch
from collections import defaultdict

from legoml.utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceTracker:
    """Thread-local performance tracking for detailed timing analysis."""

    def __init__(self):
        self.timings: Dict[str, list] = defaultdict(list)
        self.current_step_timings: Dict[str, float] = {}

    def add_timing(self, operation: str, duration: float):
        """Add a timing measurement for an operation."""
        self.timings[operation].append(duration)
        self.current_step_timings[operation] = duration

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary statistics."""
        summary = {}
        for op, times in self.timings.items():
            if times:
                summary[op] = {
                    "count": len(times),
                    "total_ms": sum(times) * 1000,
                    "avg_ms": (sum(times) / len(times)) * 1000,
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                }
        return summary

    def get_current_step_summary(self) -> Dict[str, float]:
        """Get timing for the current step only."""
        return {
            op: time_sec * 1000 for op, time_sec in self.current_step_timings.items()
        }

    def reset_current_step(self):
        """Reset current step timings."""
        self.current_step_timings = {}


# Global performance tracker instance
_perf_tracker = PerformanceTracker()


@contextmanager
def timer(operation_name: str, enabled: bool = True):
    """
    Context manager for timing code blocks.

    Args:
        operation_name: Name of the operation being timed
        enabled: Whether timing is enabled (for zero-overhead when disabled)

    Example:
        with timer("data_to_device"):
            inputs = inputs.to(device)
            targets = targets.to(device)
    """
    if not enabled:
        yield
        return

    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        _perf_tracker.add_timing(operation_name, duration)


def get_performance_summary() -> Dict[str, Dict[str, float]]:
    """Get overall performance summary across all operations."""
    return _perf_tracker.get_summary()


def get_current_step_timings() -> Dict[str, float]:
    """Get timings for the current step."""
    return _perf_tracker.get_current_step_summary()


def reset_current_step_timings():
    """Reset current step timings."""
    _perf_tracker.reset_current_step()


def log_performance_summary():
    """Log a summary of all performance timings."""
    summary = get_performance_summary()
    if not summary:
        logger.info("No performance data collected")
        return

    logger.info("=== Performance Summary ===")
    total_time = 0
    for op, stats in summary.items():
        total_time += stats["total_ms"]
        logger.info(
            f"{op}: {stats['count']} calls, "
            f"total: {stats['total_ms']:.1f}ms, "
            f"avg: {stats['avg_ms']:.1f}ms, "
            f"min: {stats['min_ms']:.1f}ms, "
            f"max: {stats['max_ms']:.1f}ms"
        )
    logger.info(f"Total measured time: {total_time:.1f}ms")


class MPSMonitor:
    """Monitor MPS (Metal Performance Shaders) performance and memory usage."""

    @staticmethod
    def is_mps_available() -> bool:
        """Check if MPS is available."""
        return torch.backends.mps.is_available()

    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """
        Get MPS memory information.
        Note: MPS doesn't have torch.mps.memory_* functions like CUDA,
        so we use system memory info instead.
        """
        if not MPSMonitor.is_mps_available():
            return {}

        # Get system memory info since MPS uses unified memory
        memory = psutil.virtual_memory()
        return {
            "system_memory_total_gb": memory.total / (1024**3),
            "system_memory_available_gb": memory.available / (1024**3),
            "system_memory_used_gb": memory.used / (1024**3),
            "system_memory_percent": memory.percent,
        }

    @staticmethod
    def log_memory_usage():
        """Log current memory usage."""
        info = MPSMonitor.get_memory_info()
        if info:
            logger.info(
                f"Memory: {info['system_memory_used_gb']:.1f}GB used, "
                f"{info['system_memory_available_gb']:.1f}GB available "
                f"({info['system_memory_percent']:.1f}% used)"
            )

    @staticmethod
    def detect_mps_fallback() -> bool:
        """
        Detect if MPS fallback is enabled.
        This checks the environment variable used by PyTorch.
        """
        return os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"


class DeviceTransferTimer:
    """Helper for timing data transfers to device."""

    def __init__(self, device: torch.device):
        self.device = device
        self.is_mps = device.type == "mps"

    @contextmanager
    def time_transfer(self, operation: str = "device_transfer"):
        """Time a device transfer operation."""
        with timer(f"{operation}_to_{self.device.type}"):
            yield

    def transfer_batch(self, batch, operation: str = "batch_to_device"):
        """Transfer a batch to device with timing."""
        with self.time_transfer(operation):
            if isinstance(batch, (list, tuple)):
                return [item.to(self.device) for item in batch]
            else:
                return batch.to(self.device)


def create_performance_context_managers(device: torch.device, enabled: bool = True):
    """
    Create a set of commonly used context managers for performance timing.

    Args:
        device: The device being used for training
        enabled: Whether timing is enabled

    Returns:
        Dict of context managers for common operations
    """
    return {
        "data_to_device": lambda: timer("data_to_device", enabled),
        "forward_pass": lambda: timer("forward_pass", enabled),
        "loss_computation": lambda: timer("loss_computation", enabled),
        "backward_pass": lambda: timer("backward_pass", enabled),
        "optimizer_step": lambda: timer("optimizer_step", enabled),
        "scheduler_step": lambda: timer("scheduler_step", enabled),
    }


def log_step_breakdown():
    """Log the breakdown of time spent in different operations for the current step."""
    timings = get_current_step_timings()
    if not timings:
        return

    total_time = sum(timings.values())
    logger.debug("Step breakdown:")
    for operation, time_ms in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        percentage = (time_ms / total_time) * 100 if total_time > 0 else 0
        logger.debug(f"  {operation}: {time_ms:.1f}ms ({percentage:.1f}%)")

    # Reset for next step
    reset_current_step_timings()

