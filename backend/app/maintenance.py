"""Housekeeping for abandoned uploads (a Kaggle session can die in the middle of a 250 MB upload).

Removed:
  - open upload sessions whose task is finished (done/failed/cancelled) or that saw no chunk for
    BACKEND_UPLOAD_STALE_HOURS, with their .part file
  - .part files with no session row, and leftover .tmp_* extraction folders, older than an hour
  - staging/<task_id>/ of tasks that ended without completing (failed/cancelled): never promoted, never needed

Runs at start-up and at most every 10 minutes from POST /worker/claim. Nothing in workspace/ is touched.
"""
from __future__ import annotations

import shutil
import time
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from .artifacts import parts_dir
from .config import Settings
from .models import Task, TaskStatus, UploadSession, utcnow
from .tasks import FINAL

ORPHAN_AGE_S = 3600
INTERVAL_S = 600


def cleanup_stale_uploads(db: Session, settings: Settings) -> dict:
    stats = {"sessions": 0, "part_files": 0, "staging_dirs": 0, "tmp_dirs": 0}
    cutoff = utcnow() - timedelta(hours=settings.upload_stale_hours)
    final_ids = set(db.scalars(select(Task.id).where(Task.status.in_(FINAL))))

    for s in db.scalars(select(UploadSession).where(UploadSession.status == "open")).all():
        if s.task_id in final_ids or s.updated_at < cutoff:
            (parts_dir(settings) / f"{s.id}.part").unlink(missing_ok=True)
            db.delete(s)
            stats["sessions"] += 1
    db.commit()

    known = set(db.scalars(select(UploadSession.id).where(UploadSession.status == "open")))
    now = time.time()
    for f in parts_dir(settings).glob("*.part"):
        if f.stem not in known and now - f.stat().st_mtime > ORPHAN_AGE_S:
            f.unlink(missing_ok=True)
            stats["part_files"] += 1

    staging = settings.data_dir / "staging"
    if staging.is_dir():
        ended = set(db.scalars(select(Task.id).where(
            Task.status.in_((TaskStatus.failed.value, TaskStatus.cancelled.value)))))
        for d in staging.iterdir():
            if d.name.startswith(".tmp_"):
                if now - d.stat().st_mtime > ORPHAN_AGE_S:
                    shutil.rmtree(d, ignore_errors=True)
                    stats["tmp_dirs"] += 1
            elif d.name in ended:
                shutil.rmtree(d, ignore_errors=True)
                stats["staging_dirs"] += 1
    return stats


def maybe_cleanup(app_state, db: Session, settings: Settings) -> dict | None:
    """Throttled entry point for request handlers."""
    last = getattr(app_state, "last_cleanup", 0.0)
    if time.time() - last < INTERVAL_S:
        return None
    app_state.last_cleanup = time.time()
    return cleanup_stale_uploads(db, settings)
