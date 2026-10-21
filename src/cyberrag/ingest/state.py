"""Small JSON checkpoint files in data/state/, plus gzip JSON helpers.
All writes go to a temp file first, so a crash never leaves a half-written file."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

from cyberrag.config import Settings


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json_gz(path: Path, data: Any) -> None:
    _atomic_write(path, gzip.compress(json.dumps(data, ensure_ascii=False).encode("utf-8"), 6))


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def state_path(settings: Settings, name: str) -> Path:
    return settings.state_dir / f"{name}.json"


def load_state(settings: Settings, name: str) -> dict[str, Any]:
    path = state_path(settings, name)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(settings: Settings, name: str, data: dict[str, Any]) -> None:
    _atomic_write(state_path(settings, name), json.dumps(data, indent=2).encode("utf-8"))


def delete_state(settings: Settings, name: str) -> None:
    state_path(settings, name).unlink(missing_ok=True)
