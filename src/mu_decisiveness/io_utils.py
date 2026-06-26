from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from threading import Lock

import numpy as np


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def load_items(path) -> list[str]:
    from .datasets import resolve_items

    return resolve_items(path)


def setup_logging(run_dir: Path, log_name: str = "run.log") -> logging.Logger:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(Path(run_dir) / log_name), logging.StreamHandler(sys.stdout)],
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("mu_decisiveness")


class JsonlAppender:
    """Append-only JSONL writer, lock-guarded for use from many async tasks."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w")  # truncate so re-runs start clean
        self._lock = Lock()

    def write(self, record: dict):
        line = json.dumps(jsonable(record), separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")

    def flush(self):
        with self._lock:
            self._fh.flush()

    def close(self):
        with self._lock:
            self._fh.close()
