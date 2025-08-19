"""Lightweight local experiment tracking for LegoML.

This module provides a minimal, optional API to track experiments locally in a
"runs/" directory. It is designed to be non-invasive and not replace MLFlow —
just a convenient default for local runs.

Example:
    from legoml.utils.track import run

    with run(name="tinycnn-mnist") as sess:
        # Prepare callbacks to write artifacts into this run's artifact folder
        ckpt_dir = sess.artifacts_dir
        sess.log_text("description", "TinyCNN on MNIST, 1 epoch smoke test")

        # ... set up trainer / dataloaders ...
        sess.log_trainer(trainer)
        sess.log_dataloader(train_loader, name="train")
        sess.log_dataloader(val_loader, name="val")

        state = trainer.fit(train_loader, val_loader)
        sess.log_scalars({**state.train_epoch}, step=state.epoch, split="train/epoch")
        if state.eval_epoch:
            sess.log_scalars({**state.eval_epoch}, step=state.epoch, split="eval/epoch")

The directory layout is:
    runs/
      <timestamp>_<shortid>_<name>/
        meta.json
        logs/
          scalars.jsonl
          events.jsonl
          objects.jsonl
        artifacts/

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import traceback
import uuid
from typing import Any, Mapping


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_uid() -> str:
    return uuid.uuid4().hex[:8]


def _safe_serialize(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable value.

    - dataclasses -> asdict
    - Mapping -> dict
    - objects -> {"type": class_name, "repr": str(obj)}
    """
    try:
        from dataclasses import is_dataclass

        if is_dataclass(obj):
            return asdict(obj)  # type: ignore
    except Exception:
        pass

    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(x) for x in obj]
    if isinstance(obj, set):
        return sorted([_safe_serialize(x) for x in obj], key=lambda x: str(x))
    if isinstance(obj, Mapping):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Fallback
    try:
        return {
            "type": type(obj).__name__,
            "repr": repr(obj),
        }
    except Exception:
        return str(obj)


def _write_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


@dataclass
class RunSession:
    """A context-managed local run tracker.

    Args:
        name: Optional human-friendly name for the run.
        base_dir: The root directory where runs are created.
        tags: Free-form labels to store in meta.
        notes: Optional description stored in meta.
    """

    name: str | None = None
    base_dir: str = "./runs"
    tags: dict[str, Any] | None = None
    notes: str | None = None

    # Runtime fields (populated on enter)
    run_id: str | None = None
    run_dir: Path | None = None
    _meta_path: Path | None = None
    _scalars_path: Path | None = None
    _events_path: Path | None = None
    _objects_path: Path | None = None

    def __enter__(self) -> "RunSession":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rid = f"{ts}_{_short_uid()}" + (f"_{self.name}" if self.name else "")
        base = Path(self.base_dir)
        base.mkdir(parents=True, exist_ok=True)

        run_dir = base / rid
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        self.run_id = rid
        self.run_dir = run_dir
        self._meta_path = run_dir / "meta.json"
        # New human-friendly formats:
        # - scalars.csv (tidy rows: time,step,epoch,key,value)
        # - events.csv (time,event,fields_json)
        # - objects/ directory with pretty JSON files per object
        self._scalars_path = run_dir / "logs" / "scalars.csv"
        self._events_path = run_dir / "logs" / "events.csv"
        self._objects_path = run_dir / "logs" / "objects"
        self._objects_path.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {
            "run_id": self.run_id,
            "name": self.name,
            "created_at": _now_iso(),
            "tags": self.tags or {},
            "notes": self.notes or "",
            "env": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "cwd": str(Path.cwd()),
            },
        }

        # Optionally enrich with git info if available
        try:
            import subprocess

            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
            meta["git"] = {
                "commit": commit,
            }
        except Exception:
            pass

        self._write_meta(meta)
        # Bind run context to structlog if configured
        try:
            from legoml.utils.logging import bind  # type: ignore

            bind(run_id=self.run_id, run_dir=str(self.run_dir))
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        status = "completed" if exc is None else "failed"
        summary: dict[str, Any] = {
            "ended_at": _now_iso(),
            "status": status,
        }
        if exc is not None:
            summary["error"] = {
                "type": getattr(exc, "__class__", type(exc)).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            }
        self._update_meta(summary)

    # Public API -----------------------------------------------------------
    @property
    def artifacts_dir(self) -> str:
        assert self.run_dir is not None
        return str(self.run_dir / "artifacts")

    def log_scalars(
        self,
        scalars: Mapping[str, float | int | None],
        step: int | None = None,
        split: str | None = None,
        epoch: int | None = None,
    ) -> None:
        """Append key-value scalars into a CSV for easy viewing.

        One row per (key, value): columns = time, step, epoch, key, value.
        If `split` is set, keys are prefixed with "{split}/" when not already.
        """
        import csv
        assert self._scalars_path is not None
        self._scalars_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._scalars_path.exists()
        now = _now_iso()
        with self._scalars_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["time", "step", "epoch", "key", "value"])
            for k, v in scalars.items():
                key = (f"{split}/{k}") if split and not k.startswith(f"{split}/") else k
                w.writerow([now, step, epoch, key, v])

    def log_event(self, name: str, **fields: Any) -> None:
        """Write a generic event to events.jsonl."""
        import csv
        assert self._events_path is not None
        self._events_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._events_path.exists()
        fields_json = json.dumps(_safe_serialize(fields), ensure_ascii=False)
        with self._events_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["time", "event", "fields_json"])
            w.writerow([_now_iso(), name, fields_json])

    def log_text(self, name: str, text: str) -> None:
        """Log a piece of text by name into objects.jsonl."""
        self._log_object(name=name, obj={"text": text})

    def log_trainer(self, trainer: Any) -> None:
        """Best-effort summary of a Trainer instance."""
        try:
            payload: dict[str, Any] = {
                "type": type(trainer).__name__,
                "max_epochs": getattr(trainer, "max_epochs", None),
                "log_every_n_steps": getattr(trainer, "log_every_n_steps", None),
            }
            strategy = getattr(trainer, "strategy", None)
            if strategy is not None:
                payload["strategy"] = {
                    "type": type(strategy).__name__,
                    "device": getattr(getattr(strategy, "device", None), "name", None)
                    or str(getattr(strategy, "device", None)),
                }
            callbacks = getattr(trainer, "callbacks", None)
            if callbacks is not None:
                payload["callbacks"] = [type(cb).__name__ for cb in callbacks]
            task = getattr(trainer, "task", None)
            if task is not None:
                payload["task"] = {"type": type(task).__name__}
            opt = getattr(trainer, "optimizer", None)
            if opt is not None:
                payload["optimizer"] = {
                    "type": type(opt).__name__,
                    "defaults": getattr(opt, "defaults", {}),
                }
            sch = getattr(trainer, "scheduler", None)
            if sch is not None:
                payload["scheduler"] = {"type": type(sch).__name__}
        except Exception as e:
            payload = {"error": f"failed to introspect trainer: {e}"}
        self._log_object(name="trainer", obj=payload)

    def log_dataloader(self, loader: Any, name: str = "train") -> None:
        """Best-effort summary of a torch DataLoader."""
        try:
            dataset = getattr(loader, "dataset", None)
            ds_len = None
            if dataset is not None:
                try:
                    ds_len = len(dataset)  # type: ignore[arg-type]
                except Exception:
                    ds_len = None
            payload: dict[str, Any] = {
                "name": name,
                "type": type(loader).__name__,
                "batch_size": getattr(loader, "batch_size", None),
                "drop_last": getattr(loader, "drop_last", None),
                "num_workers": getattr(loader, "num_workers", None),
                "pin_memory": getattr(loader, "pin_memory", None),
                "dataset": {
                    "type": type(dataset).__name__ if dataset is not None else None,
                    "length": ds_len,
                },
                "sampler": type(getattr(loader, "sampler", None)).__name__,
            }
        except Exception as e:
            payload = {"error": f"failed to introspect dataloader: {e}"}
        self._log_object(name="dataloader", obj=payload)

    def add_artifact(self, path: str, dest_name: str | None = None) -> str:
        """Copy a file into the artifacts directory and return its path.

        If `dest_name` is not provided, the original filename is used.
        """
        assert self.run_dir is not None
        src = Path(path)
        dest = Path(self.artifacts_dir) / (dest_name or src.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        return str(dest)

    def save_node(self, node: Any, filename: str = "node.json") -> str:
        """Serialize and save a model Node configuration into artifacts.

        This stores a JSON file with minimal reconstruction hints:
            {"type": ClassName, "module": module_path, "data": dataclass_dict}

        Returns the written file path.
        """
        assert self.run_dir is not None
        target = Path(self.artifacts_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Try Node.to_dict if available, else asdict/dataclass fallback
            if hasattr(node, "to_dict") and callable(getattr(node, "to_dict")):
                data = node.to_dict()  # type: ignore[attr-defined]
            else:
                from dataclasses import asdict, is_dataclass

                data = asdict(node) if is_dataclass(node) else _safe_serialize(node)
        except Exception:
            data = _safe_serialize(node)

        payload = {
            "type": type(node).__name__,
            "module": getattr(type(node), "__module__", None),
            "saved_at": _now_iso(),
            "data": data,
        }
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # Also track as an object log record for discoverability
        self._log_object(name="node", obj={"path": str(target), **payload})
        return str(target)

    # Internal -------------------------------------------------------------
    def _log_object(self, name: str, obj: Any) -> None:
        assert self._objects_path is not None
        # Write each object as a pretty JSON file for readability
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in name)
        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}.json"
        target = self._objects_path / fname
        payload = {"time": _now_iso(), "name": name, "object": _safe_serialize(obj)}
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _write_meta(self, meta: Mapping[str, Any]) -> None:
        assert self._meta_path is not None
        with self._meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _update_meta(self, patch: Mapping[str, Any]) -> None:
        assert self._meta_path is not None
        try:
            with self._meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except FileNotFoundError:
            meta = {}
        meta.update(patch)
        with self._meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def run(
    name: str | None = None,
    base_dir: str = "./runs",
    tags: dict[str, Any] | None = None,
    notes: str | None = None,
) -> RunSession:
    """Convenience factory for a RunSession context manager."""
    return RunSession(name=name, base_dir=base_dir, tags=tags, notes=notes)
