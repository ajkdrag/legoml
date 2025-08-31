import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Literal

from legoml.utils.log import get_logger

logger = get_logger(__name__)


class ExperimentSession:
    def __init__(self, run_name: str | None = None, base_dir: Path | str = "runs"):
        self.base_dir = Path(base_dir)
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = self.base_dir / self.run_name
        self.artifact_dir = self.run_dir / "artifacts"
        self.logs_dir = self.run_dir / "logs"
        self._metadata = {}
        self._is_active = False

    def __enter__(self) -> "ExperimentSession":
        self._setup_dirs()
        self._is_active = True
        logger.info(
            "Started experiment session",
            run_name=self.run_name,
            run_dir=str(self.run_dir),
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._metadata:
            self._save_metadata()
        self._is_active = False
        logger.info("Ended experiment session", run_name=self.run_name)

    def _setup_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

    def _save_metadata(self) -> None:
        metadata_file = self.run_dir / "metadata.json"
        with metadata_file.open("w") as f:
            json.dump(self._metadata, f, indent=2)

    def get_run_dir(self) -> Path:
        return self.run_dir

    def get_artifact_dir(self) -> Path:
        return self.artifact_dir

    def get_logs_dir(self) -> Path:
        return self.logs_dir

    def log_text(self, name: str, text: str) -> None:
        if not self._is_active:
            logger.warning("Session not active, text not logged", name=name)
            return

        text_file = self.logs_dir / f"{name}.txt"
        with text_file.open("a") as f:
            f.write(text + "\n")

        logger.debug("Logged text", name=name)

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._is_active:
            logger.warning("Session not active, params not logged")
            return

        self._metadata.setdefault("params", {}).update(params)
        logger.debug("Logged params", count=len(params))

    def save_artifact(
        self, obj: Any, name: str, format: Literal["json", "torch"] = "json"
    ) -> Path:
        if not self._is_active:
            logger.warning("Session not active, artifact not saved", name=name)
            return self.artifact_dir / f"{name}.{format}"

        if format == "json":
            artifact_path = self.artifact_dir / f"{name}.json"
            with artifact_path.open("w") as f:
                json.dump(obj, f, indent=2, default=str)
        else:
            import torch

            artifact_path = self.artifact_dir / f"{name}.pth"
            torch.save(obj, artifact_path)

        logger.info("Saved artifact", name=name, path=str(artifact_path))
        return artifact_path


@contextmanager
def run(
    run_name: str | None = None, base_dir: Path | str = "runs"
) -> Generator[ExperimentSession, None, None]:
    session = ExperimentSession(run_name, base_dir)
    with session as sess:
        yield sess
