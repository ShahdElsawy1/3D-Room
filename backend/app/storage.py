"""Filesystem layout. One folder per project, identical to Person A's pipeline contract
(ML_Work/common/config.py), so `run_pipeline.py --root <workspace> --project <project_id>` and all
Week 2 code read the same paths:

    <data_dir>/workspace/<project_id>/
        video/input.<ext>          uploaded original (never modified)
        frames/ poses/ scene/ gs/ export/ metrics/ manifest.json      (written back by the worker, M3/M4)
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from .config import Settings


def project_dir(settings: Settings, project_id: str) -> Path:
    return settings.workspace_dir / project_id


def resolve_in_project(settings: Settings, project_id: str, rel_path: str) -> Path:
    """Resolve a stored relative path, refusing anything that escapes the project folder."""
    base = project_dir(settings, project_id).resolve()
    p = (base / rel_path).resolve()
    if base != p and base not in p.parents:
        raise ValueError(f"path escapes project folder: {rel_path}")
    return p


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def delete_project_dir(settings: Settings, project_id: str) -> None:
    d = project_dir(settings, project_id)
    if d.exists():
        shutil.rmtree(d)
