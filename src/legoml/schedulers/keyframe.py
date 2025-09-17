import math
from dataclasses import dataclass, field
from typing import (
    Callable,
    List,
    Literal,
    Protocol,
    Sequence,
    TypeAlias,
    TypeGuard,
    Union,
    cast,
    runtime_checkable,
)

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from legoml.utils.log import get_logger

logger = get_logger(__name__)


# =========================
# Public typing surface
# =========================


@dataclass(frozen=True)
class Keyframe:
    position: Union[float, int, Literal["end"]]
    lr: Union[float, List[float]]  # per-param-group list allowed


@dataclass
class InternKeyframe:
    position: float
    lr: float


ScheduleEntry: TypeAlias = Union[InternKeyframe, "TransitionFn"]
Schedule: TypeAlias = List[ScheduleEntry]


@runtime_checkable
class TransitionFn(Protocol):
    def __call__(
        self,
        last_lr: float,
        start_frame: InternKeyframe,
        end_frame: InternKeyframe,
        position: float,
        scheduler: "KeyframeLR",
        *,
        group_idx: int,
    ) -> float: ...


FrameDef: TypeAlias = Union[Keyframe, TransitionFn]
FrameSeq: TypeAlias = Sequence[FrameDef]
FramesArg: TypeAlias = Union[FrameSeq, Sequence[FrameSeq]]


def _is_seq_of_frame_seqs(x: object) -> TypeGuard[Sequence[FrameSeq]]:
    if not isinstance(x, Sequence) or len(x) == 0:
        return False
    first = x[0]
    if not isinstance(first, Sequence):
        return False
    if len(first) == 0:
        return True
    f0 = first[0]
    return isinstance(f0, Keyframe) or callable(f0)


# =========================
# Built-in transitions
# =========================


@dataclass(frozen=True)
class ExponentialDecay:
    total_steps: int
    gamma: float

    def __call__(
        self,
        last_lr: float,
        start_frame: InternKeyframe,
        end_frame: InternKeyframe,
        position: float,
        scheduler: "KeyframeLR",
        *,
        group_idx: int,
    ) -> float:
        steps = (
            position
            if scheduler.units == "steps"
            else position * float(self.total_steps)
        )
        return start_frame.lr * (self.gamma ** float(steps))


@dataclass(frozen=True)
class LinearInterpolation:
    def __call__(
        self,
        last_lr: float,
        start_frame: InternKeyframe,
        end_frame: InternKeyframe,
        position: float,
        scheduler: "KeyframeLR",
        *,
        group_idx: int,
    ) -> float:
        span = end_frame.position - start_frame.position
        pct = 0.0 if span == 0.0 else (position - start_frame.position) / span
        return (1.0 - pct) * start_frame.lr + pct * end_frame.lr


@dataclass(frozen=True)
class CosineInterpolation:
    def __call__(
        self,
        last_lr: float,
        start_frame: InternKeyframe,
        end_frame: InternKeyframe,
        position: float,
        scheduler: "KeyframeLR",
        *,
        group_idx: int,
    ) -> float:
        span = end_frame.position - start_frame.position
        pct = 0.0 if span == 0.0 else (position - start_frame.position) / span
        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * pct))
        return cosine_weight * start_frame.lr + (1.0 - cosine_weight) * end_frame.lr


@dataclass(kw_only=True)
class ReduceLROnMetricPlateau:
    best_fn: Callable[..., float]
    best_init: float = -float("inf")
    patience: int
    factor: float
    min_lr: float
    threshold: float
    threshold_mode: str
    state: dict = field(default_factory=dict)

    def __call__(
        self,
        last_lr: float,
        start_frame: InternKeyframe,
        end_frame: InternKeyframe,
        position: float,
        scheduler: "KeyframeLR",
        *,
        group_idx: int,
    ) -> float:
        # init per-scheduler, per-group state
        if group_idx not in self.state:
            self.state[group_idx] = {
                "best": self.best_init,
                "bad_count": 0,
            }

        st = self.state[group_idx]
        metric = self.best_fn()
        improved = (
            (metric > st["best"] + self.threshold)
            if self.threshold_mode == "abs"
            else (metric > st["best"] * (1.0 + self.threshold))
        )

        if improved:
            st["best"] = metric
            st["bad_count"] = 0
            return last_lr

        # not improved
        st["bad_count"] += 1
        logger.warning(
            "Metric didn't improve.",
            state=st,
            group_idx=group_idx,
        )

        if st["bad_count"] > self.patience:
            new_lr = max(last_lr * self.factor, self.min_lr)
            logger.warning(
                "Lost patience. Updating lr",
                state=st,
                lr=new_lr,
                group_idx=group_idx,
            )
            st["bad_count"] = 0
            return new_lr

        return last_lr


# =========================
# Scheduler
# =========================


class KeyframeLR(LRScheduler):
    """
    Frames allowed per group:
      - Keyframe(position, lr)
      - Transition.LINEAR / Transition.COSINE
      - Callable(last_lr, start_frame, end_frame, position, scheduler, *,
                 group_idx) -> lr
    """

    def __init__(
        self,
        optimizer: Optimizer,
        frames: FramesArg,
        end: float,
        units: Literal["percent", "steps"] = "steps",
    ) -> None:
        self.end = float(end)
        self.units = units
        self.optimizer = optimizer
        self.last_step = -1

        n_groups = len(optimizer.param_groups)
        frames_per_group = self._normalize_frames(frames, n_groups)
        if len(frames_per_group) != n_groups:
            raise ValueError(
                f"Expected {n_groups} schedules for {n_groups} groups, "
                "got {len(frames_per_group)}."
            )

        self.schedules: List[Schedule] = [
            self._parse_one_schedule(
                frames_per_group[i],
                i,
            )
            for i in range(n_groups)
        ]
        # In step 0, this will be overriden by first keyframe lrs
        self.last_lrs: List[float] = [0.0] * n_groups

    def _normalize_frames(
        self, frames: FramesArg, n_groups: int
    ) -> List[List[FrameDef]]:
        if _is_seq_of_frame_seqs(frames):
            return [list(inner) for inner in cast(Sequence[FrameSeq], frames)]
        single = list(cast(FrameSeq, frames))
        return [single for _ in range(n_groups)]

    def _parse_one_schedule(
        self, user_frames: Sequence[FrameDef], group_idx: int
    ) -> Schedule:
        frames: Schedule = []
        end_pos = self.end if self.units == "steps" else 1.0

        unpacked: List[ScheduleEntry] = []
        keyframes: List[InternKeyframe] = []

        for f in user_frames:
            if isinstance(f, Keyframe):
                pos = float(end_pos if f.position == "end" else f.position)
                lr = float(f.lr[group_idx] if isinstance(f.lr, (list, tuple)) else f.lr)
                kf = InternKeyframe(position=pos, lr=lr)
                unpacked.append(cast(ScheduleEntry, kf))
                keyframes.append(kf)
            elif isinstance(f, TransitionFn):
                unpacked.append(cast(ScheduleEntry, f))
            else:
                raise TypeError(f"Invalid frame type: {type(f)}")

        if not keyframes:
            return frames

        sorted_keyframes: List[InternKeyframe] = sorted(
            keyframes,
            key=lambda d: d.position,
        )

        last_kf: InternKeyframe | None = None
        for i, kf in enumerate(sorted_keyframes):
            if i == 0:
                if kf.position > 0.0:
                    frames.append(InternKeyframe(position=0.0, lr=kf.lr))
                    # TODO: append default linear transition
                frames.append(kf)
            else:
                # default transition
                transition = LinearInterpolation()

                # search first transition between last_kf and kf in the original order
                start_idx = unpacked.index(cast(ScheduleEntry, last_kf))
                end_idx = unpacked.index(cast(ScheduleEntry, kf))
                for j in range(start_idx + 1, end_idx):
                    cur = unpacked[j]
                    if isinstance(cur, TransitionFn):
                        transition = cur
                        break

                frames.append(transition)
                frames.append(kf)
            last_kf = kf

        if last_kf is not None and last_kf.position < float(end_pos):
            frames.append(LinearInterpolation())
            frames.append(InternKeyframe(position=float(end_pos), lr=last_kf.lr))

        return frames

    def _lr_for_schedule(
        self, position: float, schedule: Schedule, last_lr: float, group_idx: int
    ) -> float:
        if not schedule:
            return last_lr

        for e in schedule:
            if isinstance(e, InternKeyframe) and e.position == position:
                return e.lr

        for i in range(0, len(schedule) - 2, 2):
            s = cast(InternKeyframe, schedule[i])
            fn = cast(TransitionFn, schedule[i + 1])
            e = cast(InternKeyframe, schedule[i + 2])
            if s.position <= position < e.position:
                return fn(
                    last_lr,
                    s,
                    e,
                    position,
                    self,
                    group_idx=group_idx,
                )

        last_kf = cast(InternKeyframe, schedule[-1])
        if position >= last_kf.position:
            return last_kf.lr

        return last_lr

    def get_lr(self) -> List[float]:
        if self.units == "steps":
            position = float(self.last_step)
        else:
            position = (self.last_step / self.end) if self.end > 0.0 else 1.0

        new_lrs: List[float] = []
        for i, sched in enumerate(self.schedules):
            lr = self._lr_for_schedule(
                position,
                sched,
                self.last_lrs[i],
                group_idx=i,
            )
            new_lrs.append(lr)
        self.last_lrs = new_lrs
        return new_lrs

    def step(self, epoch=None):
        self.last_step += 1
        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group["lr"] = lr


# =========================
# Schedule helpers
# =========================


def create_linear_schedule(start_lr: float, end_lr: float) -> List[FrameDef]:
    return [
        Keyframe(0, start_lr),
        LinearInterpolation(),
        Keyframe(
            "end",
            end_lr,
        ),
    ]


def create_exponential_schedule(
    start_lr: float, end_lr: float, total_steps: int
) -> List[FrameDef]:
    if start_lr <= 0.0 or end_lr <= 0.0:
        raise ValueError("Exponential schedule requires positive start and end LRs.")
    gamma = (end_lr / start_lr) ** (1.0 / float(total_steps))
    return [
        Keyframe(0, start_lr),
        ExponentialDecay(total_steps=total_steps, gamma=gamma),
        Keyframe("end", end_lr),
    ]


def create_cyclic_schedule(
    base_lr: float, max_lr: float, step_size: int, total_steps: int
) -> List[FrameDef]:
    frames: List[FrameDef] = []
    cur = 0
    while cur < total_steps:
        frames.append(Keyframe(cur, base_lr))
        frames.append(LinearInterpolation())
        cur += step_size
        if cur > total_steps:
            break
        frames.append(Keyframe(cur, max_lr))
        frames.append(LinearInterpolation())
        cur += step_size
    frames.append(Keyframe(total_steps, base_lr))
    return frames
