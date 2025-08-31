import os
from pathlib import Path
from typing import Optional
import torch
from torch.profiler import profile, ProfilerActivity, schedule

from legoml.core.context import Context
from legoml.core.callback import Callback, implements
from legoml.core.state import EngineState
from legoml.utils.log import get_logger

logger = get_logger(__name__)


@implements("on_engine_start", "on_step_start", "on_step_end", "on_engine_end")
class ProfilerCallback(Callback):
    def __init__(
        self,
        output_dir: str | Path = "./profiler_traces",
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 2,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        activities: Optional[list[ProfilerActivity]] = None,
    ):
        """
                P
                yT
                or
                ch
                pr
                of
                il
                er
                ca
                ll
                ba
                ck
                fo
                r
                pe
                rf
                or
                ma
                nc
                e
        analysis.

                Args:
                    output_dir: Directory to save profiler traces
                    wait: Number of steps to skip before profiling
                    warmup: Number of warmup steps
                    active: Number of active profiling steps
                    repeat: Number of cycles to repeat profiling
                    record_shapes: Record tensor shapes
                    profile_memory: Enable memory profiling
                    with_stack: Record stack traces (slower but more detailed)
                    activities: List of activities to profile (defaults to CPU and CUDA/MPS)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default activities - detect MPS vs CUDA
        if activities is None:
            activities = [ProfilerActivity.CPU]
            if torch.backends.mps.is_available():
                # Note: MPS profiling might have limitations
                logger.info(
                    "MPS detected - CPU profiling enabled. MPS profiling support is limited."
                )
            elif torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)

        self.profiler_schedule = schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        )

        self.activities = activities
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack

        self.profiler: Optional[profile] = None
        self.step_count = 0

    def on_engine_start(self, context: Context, state: EngineState):
        trace_file = self.output_dir / f"trace_epoch_{state.epoch}.json"

        self.profiler = profile(
            activities=self.activities,
            schedule=self.profiler_schedule,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            on_trace_ready=lambda p: p.export_chrome_trace(str(trace_file)),
        )

        self.profiler.start()
        logger.info(f"Started profiler - traces will be saved to {trace_file}")

    def on_step_start(self, context: Context, state: EngineState, batch):
        if self.profiler is not None:
            self.profiler.step()

    def on_step_end(self, context: Context, state: EngineState, batch):
        self.step_count += 1

        # Simple profiling status logging without accessing private attributes
        if self.step_count <= 10 or self.step_count % 50 == 0:
            logger.debug(f"Profiler step {self.step_count}: profiling active")

    def on_engine_end(self, context: Context, state: EngineState):
        if self.profiler is not None:
            self.profiler.stop()

            # Export additional profiling data
            profile_dir = self.output_dir / f"profile_epoch_{state.epoch}"
            profile_dir.mkdir(exist_ok=True)

            # Export table view
            table_file = profile_dir / "profile_table.txt"
            with open(table_file, "w") as f:
                f.write(
                    self.profiler.key_averages().table(
                        sort_by="cpu_time_total", row_limit=50
                    )
                )

            # Export memory summary if memory profiling enabled
            if self.profile_memory:
                memory_file = profile_dir / "memory_profile.txt"
                with open(memory_file, "w") as f:
                    f.write(
                        self.profiler.key_averages().table(
                            sort_by="cpu_memory_usage", row_limit=20
                        )
                    )

            logger.info(f"Profiler stopped. Results saved to {profile_dir}")
            self.profiler = None
