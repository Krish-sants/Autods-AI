import json
from pathlib import Path
from typing import Any

from app.config import get_settings

settings = get_settings()


def run_dir(run_id: str) -> Path:
    path = settings.runs_dir / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path(run_id: str, name: str) -> Path:
    return run_dir(run_id) / name


def save_json(run_id: str, name: str, data: Any) -> Path:
    path = artifact_path(run_id, name)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def load_json(run_id: str, name: str) -> Any:
    path = artifact_path(run_id, name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_bytes(run_id: str, name: str, data: bytes) -> Path:
    path = artifact_path(run_id, name)
    path.write_bytes(data)
    return path


def save_text(run_id: str, name: str, text: str) -> Path:
    path = artifact_path(run_id, name)
    path.write_text(text, encoding="utf-8")
    return path


def artifact_exists(run_id: str, name: str) -> bool:
    return artifact_path(run_id, name).exists()


def list_artifacts(run_id: str) -> list[str]:
    d = run_dir(run_id)
    return sorted(p.name for p in d.iterdir() if p.is_file())
