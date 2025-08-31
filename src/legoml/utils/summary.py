import math
from collections import OrderedDict
from dataclasses import dataclass, field
from legoml.utils.log import get_logger
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

log = get_logger(__name__)

UNITS: Tuple[str, ...] = (" ", "K", "M", "B", "T")
UNKNOWN_SHAPE = "?"


def human_count(n: int) -> str:
    """Return a compact, human-readable count with unit.

    Examples:
        >>> human_count(0)
        '0  '
        >>> human_count(1234)
        '1.2 K'
        >>> human_count(2_000_000)
        '2.0 M'
    """
    if n == 0:
        return "0  "
    if n < 0:
        raise ValueError("Count cannot be negative")

    group = int(math.floor(math.log10(n) / 3))
    group = max(0, min(group, len(UNITS) - 1))
    unit = UNITS[group]
    scaled = n / (1000**group)
    if group == 0:
        return f"{int(scaled):,d} {unit}"
    return f"{scaled:,.1f} {unit}"


def infer_shape(x: Any) -> Union[str, List]:
    """Recursively infer a human-friendly "shape" from common containers/Tensors."""
    if hasattr(x, "shape"):
        # torch.Size -> list for readability
        return list(getattr(x, "shape"))
    if isinstance(x, (list, tuple)):
        return [infer_shape(el) for el in x]
    return UNKNOWN_SHAPE


def _is_uninitialized_param(p: Tensor) -> bool:
    # Lazy modules keep UninitializedParameter until first use
    if isinstance(p, nn.parameter.UninitializedParameter):
        log.warning(
            "Found UninitializedParameter. Parameter counts/sizes may be underestimated."
        )
        return True
    return False


def count_parameters(module: nn.Module) -> int:
    """Total number of *initialized* parameters in a module."""
    total = 0
    for p in module.parameters():
        if not _is_uninitialized_param(p):
            total += p.numel()
    return total


def parameter_size_bytes(model: nn.Module) -> int:
    """Accurately estimate total parameter memory in bytes.

    Uses each parameter's actual dtype via `element_size()`; skips uninitialized.
    """
    size = 0
    for p in model.parameters():
        if not _is_uninitialized_param(p):
            size += p.element_size() * p.numel()
    return size


@dataclass
class LayerInfo:
    name: str
    module: nn.Module
    layer_type: str = field(init=False)
    in_shape: Union[str, List] = UNKNOWN_SHAPE
    out_shape: Union[str, List] = UNKNOWN_SHAPE

    def __post_init__(self) -> None:
        self.layer_type = self.module.__class__.__name__

    @property
    def param_count(self) -> int:
        return count_parameters(self.module)


# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------


def _collect_modules(
    model: nn.Module,
    max_depth: int,
    include_parents: bool,
) -> List[Tuple[str, nn.Module]]:
    """Return list of (name, module) pairs according to depth rules.

    Depth semantics:
      - 0: no layers
      - 1: direct children
      - -1 or >1: named_modules with optional parent filtering
    """
    if max_depth == 0:
        return []
    if max_depth == 1:
        return list(model.named_children())

    # -1 or > 1
    named = list(model.named_modules())
    if include_parents and named:
        named = named[1:]  # drop the root module itself

    if max_depth > 1:
        named = [(n, m) for (n, m) in named if n.count(".") < max_depth]
    return named


def _register_io_hooks(
    layers: Mapping[str, LayerInfo],
    named_modules: Iterable[Tuple[str, nn.Module]],
) -> List[RemovableHandle]:
    """Attach forward hooks to capture input/output shapes for given layers."""

    def make_hook(info: LayerInfo):
        def hook(_module: nn.Module, inp: Tuple[Any, ...], out: Any) -> None:
            # Normalize first positional input if present
            if isinstance(inp, (list, tuple)) and len(inp) == 1:
                inp = inp[0]
            info.in_shape = infer_shape(inp)
            info.out_shape = infer_shape(out)

        return hook

    handles: List[RemovableHandle] = []
    for name, module in named_modules:
        if name in layers:
            handles.append(module.register_forward_hook(make_hook(layers[name])))
    return handles


def _safe_forward(model: nn.Module, example: Any, **kwargs: Any) -> None:
    """Run a forward pass to trigger hooks, swallowing errors with a warning."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if isinstance(example, (list, tuple)):
                model(*example, **kwargs)
            elif isinstance(example, Mapping):
                model(**example, **kwargs)
            else:
                model(example, **kwargs)
    except Exception as exc:  # noqa: BLE001
        log.warning("Forward pass failed for summary capture: %s", exc)
    finally:
        model.train(was_training)


def _format_table(
    *,
    total_params: int,
    trainable_params: int,
    param_bytes: int,
    layers: "OrderedDict[str, LayerInfo]",
    include_io: bool,
) -> str:
    """Build a pretty summary table string."""

    def col(col_name: str) -> List[str]:
        if col_name == "Name":
            return list(layers.keys())
        if col_name == "Type":
            return [li.layer_type for li in layers.values()]
        if col_name == "Params":
            return [human_count(li.param_count) for li in layers.values()]
        if col_name == "In":
            return [str(li.in_shape) for li in layers.values()]
        if col_name == "Out":
            return [str(li.out_shape) for li in layers.values()]
        raise KeyError(col_name)

    columns: Tuple[str, ...] = ("Name", "Type", "Params")
    if include_io:
        columns += ("In", "Out")

    data: Dict[str, List[str]] = {name: col(name) for name in columns}

    # Column widths
    widths: Dict[str, int] = {}
    for name, values in data.items():
        widths[name] = (
            max(len(name), *(len(str(v)) for v in values)) if values else len(name)
        )

    header = " | ".join(f"{n:<{widths[n]}}" for n in columns)
    line = "-" * len(header)

    lines: List[str] = [line, header, line]
    for i in range(len(next(iter(data.values()), []))):
        row = [data[name][i] for name in columns]
        lines.append(
            " | ".join(f"{cell:<{widths[name]}}" for cell, name in zip(row, columns))
        )
    lines.append(line)

    # Footer
    non_trainable = total_params - trainable_params
    mib = param_bytes / (1024**2)

    def footer_row(left: str, right: str) -> str:
        return f"{left:<14} {right}"

    lines.extend(
        [
            footer_row(human_count(trainable_params), "Trainable params"),
            footer_row(human_count(non_trainable), "Non-trainable params"),
            footer_row(human_count(total_params), "Total params"),
            footer_row(f"{mib:,.3f}", "Total estimated parameter size (MiB)"),
        ]
    )

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def summarize_model(
    model: nn.Module,
    example_input: Any,
    *,
    depth: int = 1,
    capture_io: bool = True,
    include_parent_layers: bool = True,
    print_result: bool = True,
    **forward_kwargs: Any,
) -> str:
    """Create a readable summary for a PyTorch `nn.Module`.

    Args:
        model: The module to summarize.
        example_input: Sample input used to probe I/O shapes (tuple/list/dict/tensor).
        depth: 0=no layers, 1=children, -1=all, >1=limit nesting depth.
        capture_io: If True, records input/output shapes via forward hooks.
        include_parent_layers: When depth != 1, include non-leaf parents.
        print_result: If True, prints the summary to stdout.
        **forward_kwargs: Extra kwargs forwarded into the model during probing.

    Returns:
        The formatted summary string.
    """
    named = _collect_modules(
        model, max_depth=depth, include_parents=include_parent_layers
    )
    layers: "OrderedDict[str, LayerInfo]" = OrderedDict(
        (n, LayerInfo(n, m)) for n, m in named
    )

    handles: List[RemovableHandle] = []
    if capture_io and layers:
        handles = _register_io_hooks(layers, named)

    # Trigger hooks
    _safe_forward(model, example_input, **forward_kwargs)

    # Always remove hooks
    for h in handles:
        h.remove()

    # Aggregate totals
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_param_bytes = parameter_size_bytes(model)

    result = _format_table(
        total_params=total_params,
        trainable_params=trainable_params,
        param_bytes=total_param_bytes,
        layers=layers,
        include_io=capture_io,
    )

    if print_result:
        print(result)
    return result


if __name__ == "__main__":

    class SimpleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(16 * 16 * 16, 128),  # For 32x32 input
                nn.BatchNorm1d(128),
            )
            self.classifier = nn.Linear(128, 10)

        def forward(self, x: Tensor) -> Tensor:
            features = self.net(x)
            return self.classifier(features)

    model = SimpleModel()
    sample = torch.randn(16, 3, 32, 32)

    log.info("\n--- Depth=1 (children) ---")
    summarize_model(model, sample, depth=1)

    log.info("\n--- Depth=-1 (all) ---")
    summarize_model(model, sample, depth=-1)

    log.info("\n--- Depth=2 ---")
    summarize_model(model, sample, depth=2)
