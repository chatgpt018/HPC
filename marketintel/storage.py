"""Small durable on-disk document and job store for the local prototype."""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def project_path(value: str | Path) -> Path:
    """Resolve a configurable path from the extracted project folder."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


def data_directory() -> Path:
    """Find the data folder without creating it (safe for diagnostics)."""
    return project_path(os.environ.get('MARKETINTEL_DATA_DIR', 'storage'))


def home() -> Path:
    path = data_directory()
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False, suffix='.tmp') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        tmp = Path(handle.name)
    tmp.replace(path)


def records() -> list[dict]:
    return read_json(home() / 'documents.json', [])


def chunks() -> list[dict]:
    return read_json(home() / 'chunks.json', [])
