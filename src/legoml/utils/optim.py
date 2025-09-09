import re
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Sequence,
    Tuple,
)

import torch.nn as nn

from legoml.nn.norm import GRN

NORM_CLASSES: Tuple[type, ...] = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LocalResponseNorm,
    GRN,
)

Predicate = Callable[[str, nn.Parameter, nn.Module], bool]
Updater = Callable[[Mapping[str, Any], str, nn.Parameter, nn.Module], Mapping[str, Any]]


@dataclass(frozen=True)
class Rule:
    predicate: Predicate
    updater: Updater


def _param_owner_map(root: nn.Module) -> Dict[str, nn.Module]:
    out: Dict[str, nn.Module] = {}
    for mod_name, mod in root.named_modules():
        for p_name, _ in mod.named_parameters(recurse=False):
            full = f"{mod_name}.{p_name}" if mod_name else p_name
            out[full] = mod
    return out


def _dedup_params(
    params: Iterable[Tuple[str, nn.Parameter]],
) -> List[Tuple[str, nn.Parameter]]:
    seen: set[int] = set()
    out: List[Tuple[str, nn.Parameter]] = []
    for n, p in params:
        if not p.requires_grad or not p.is_floating_point():
            continue
        i = id(p)
        if i in seen:
            continue
        seen.add(i)
        out.append((n, p))
    return out


def build_param_groups(
    model: nn.Module,
    base: Mapping[str, Any],
    rules: Sequence[Rule] = (),
) -> List[Dict[str, Any]]:
    owners = _param_owner_map(model)
    named = list(model.named_parameters())
    named = _dedup_params(named)
    buckets: Dict[Tuple[Tuple[str, Any], ...], List[nn.Parameter]] = {}
    for name, p in named:
        mod = owners.get(name, model)
        cfg: Dict[str, Any] = dict(base)
        for r in rules:
            if r.predicate(name, p, mod):
                cfg.update(r.updater(cfg, name, p, mod))
        key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
        buckets.setdefault(key, []).append(p)
    groups = [{"params": ps, **dict(key)} for key, ps in buckets.items() if len(ps)]
    return groups


# ---------- Useful rule factories ----------


def no_weight_decay_on_bias() -> Rule:
    return Rule(
        predicate=lambda n, p, m: n.endswith(".bias"),
        updater=lambda cfg, n, p, m: {"weight_decay": 0.0},
    )


def no_weight_decay_on_norm(norms: Tuple[type, ...] = NORM_CLASSES) -> Rule:
    return Rule(
        predicate=lambda n, p, m: isinstance(m, norms),
        updater=lambda cfg, n, p, m: {"weight_decay": 0.0},
    )


def no_weight_decay_on_1d() -> Rule:
    return Rule(
        predicate=lambda n, p, m: p.ndim == 1,
        updater=lambda cfg, n, p, m: {"weight_decay": 0.0},
    )


def match_name_regex(pattern: str, **overrides: Any) -> Rule:
    rx = re.compile(pattern)
    return Rule(
        predicate=lambda n, p, m: bool(rx.search(n)),
        updater=lambda cfg, n, p, m: overrides,
    )


def match_prefix(prefixes: Sequence[str], **overrides: Any) -> Rule:
    prefixes = tuple(prefixes)
    return Rule(
        predicate=lambda n, p, m: n.startswith(prefixes),
        updater=lambda cfg, n, p, m: overrides,
    )


def module_is(classes: Tuple[type, ...], **overrides: Any) -> Rule:
    return Rule(
        predicate=lambda n, p, m: isinstance(m, classes),
        updater=lambda cfg, n, p, m: overrides,
    )


def layerwise_lr_decay(
    depth: Callable[[str], int], gamma: float, base_lr_key: str = "lr"
) -> Rule:
    return Rule(
        predicate=lambda n, p, m: True,
        updater=lambda cfg, n, p, m: {
            base_lr_key: cfg[base_lr_key] * (gamma ** depth(n))
        },
    )


# ---------- Sensible defaults ----------


def default_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    no_wd_bias: bool = True,
    no_wd_norm: bool = True,
    no_wd_1d: bool = True,
    no_wd_layer_scale: bool = True,
) -> List[Dict[str, Any]]:
    rules: List[Rule] = []
    if no_wd_bias:
        rules.append(no_weight_decay_on_bias())
    if no_wd_norm:
        rules.append(no_weight_decay_on_norm())
    if no_wd_1d:
        rules.append(no_weight_decay_on_1d())
    if no_wd_layer_scale:
        from legoml.nn.ops import LayerScale

        _rule = module_is((LayerScale,), lr=0.1, weight_decay=0.0)
        rules.append(_rule)
    return build_param_groups(
        model, base={"lr": lr, "weight_decay": weight_decay}, rules=rules
    )


# ------------ Helpers -----------


def _param_name_map(model: nn.Module) -> Dict[int, str]:
    return {id(p): n for n, p in model.named_parameters()}


def print_param_groups(
    model: nn.Module,
    groups: Iterable[Mapping],
    max_names: int | None = None,
) -> None:
    id2name = _param_name_map(model)
    for i, g in enumerate(groups):
        opts = {k: v for k, v in g.items() if k != "params"}
        names = [
            id2name.get(id(p), f"<unmapped:{tuple(p.shape)}>") for p in g["params"]
        ]
        names.sort()
        hdr = f"[group {i}] n={len(names)} " + " ".join(
            f"{k}={v}" for k, v in sorted(opts.items())
        )
        print(hdr)
        shown = names if max_names is None else names[:max_names]
        for n in shown:
            print(f"  - {n}")
        if max_names is not None and len(names) > max_names:
            print(f"  ... (+{len(names) - max_names} more)")
