from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_git_info() -> dict[str, Any]:
    def run(cmd: list[str]) -> str | None:
        try:
            return (
                subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            )
        except Exception:
            return None

    sha = run(["git", "rev-parse", "HEAD"]) or "unknown"
    status = run(["git", "status", "--porcelain"]) or ""
    return {"sha": sha, "dirty": bool(status)}


@dataclass
class ExperimentRun(AbstractContextManager["ExperimentRun"]):
    name: str
    base_dir: str | os.PathLike[str] = "runs"

    exp_id: str = field(init=False)
    run_dir: Path = field(init=False)
    index: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.exp_id = f"{self.name}-{ts}"
        self.run_dir = Path(self.base_dir) / self.exp_id

    # Context manager API
    def __enter__(self) -> "ExperimentRun":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "exp_id": self.exp_id,
            "created_at": _now_iso(),
            "git": _safe_git_info(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # Copy dependency lock/context for reproducibility if present
        for fname in ("pyproject.toml", "uv.lock"):
            if Path(fname).exists():
                shutil.copy(fname, self.run_dir / fname)

        self.index = {"stages": []}
        self._write_index()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self._write_index()

    # Index helpers
    def _write_index(self) -> None:
        (self.run_dir / "index.json").write_text(json.dumps(self.index, indent=2))

    # Stage management
    def stage_dir(self, stage: str) -> Path:
        return self.run_dir / "stages" / stage

    def start_stage(self, stage: str, *, config: dict[str, Any] | None = None) -> Path:
        sdir = self.stage_dir(stage)
        (sdir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (sdir / "artifacts").mkdir(parents=True, exist_ok=True)

        # Persist stage config
        if config is None:
            config = {}
        (sdir / "config.json").write_text(json.dumps(config, indent=2))

        # Create empty metrics file (JSONL)
        (sdir / "metrics.jsonl").touch(exist_ok=True)

        # Update index
        entry = {
            "name": stage,
            "path": str(sdir),
            "created_at": _now_iso(),
            "config": config,
        }
        # Replace if exists
        existing = [e for e in self.index["stages"] if e.get("name") == stage]
        if existing:
            self.index["stages"] = [
                e for e in self.index["stages"] if e.get("name") != stage
            ]
        self.index["stages"].append(entry)
        self._write_index()
        return sdir

    def log_metrics(self, stage: str, payload: dict[str, Any]) -> None:
        sdir = self.stage_dir(stage)
        payload = {"timestamp": _now_iso(), **payload}
        with (sdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(payload) + "\n")

    def end_stage(self, stage: str, *, summary: dict[str, Any] | None = None) -> None:
        sdir = self.stage_dir(stage)
        if summary is None:
            summary = {}
        (sdir / "summary.json").write_text(json.dumps(summary, indent=2))
        # Update index with summary
        for e in self.index.get("stages", []):
            if e.get("name") == stage:
                e["summary"] = summary
                e["ended_at"] = _now_iso()
        self._write_index()
