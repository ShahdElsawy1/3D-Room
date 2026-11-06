"""Task queue logic for the pull-based worker (spec 4.3). No Celery/Redis: the DB row IS the queue.

Life cycle:
    pending --claim--> running --complete--> done
                          |----fail(retryable, attempts left)--> pending
                          |----fail / attempts used up---------> failed
                          |----lease expired (worker vanished)--> pending | failed
    pending|running --user cancel--> cancelled   (a running worker learns it from heartbeat/progress replies)

The lease is renewed by every progress post and every heartbeat. Kaggle sessions die without warning, so a
silent worker must not keep a task "running" forever.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from .models import Task, TaskStatus, utcnow

ACTIVE = (TaskStatus.pending.value, TaskStatus.running.value)
FINAL = (TaskStatus.done.value, TaskStatus.failed.value, TaskStatus.cancelled.value)

# Stages shown in the progress bar for a reconstruction (spec 4.4), with the overall % range of each.
# Must match ML_Work/common/progress.py STAGE_RANGES, which the pipeline uses to report.
RECONSTRUCT_STAGES = [
    {"name": "extracting frames", "start": 0, "end": 15},
    {"name": "estimating poses", "start": 15, "end": 45},
    {"name": "training", "start": 45, "end": 90},
    {"name": "exporting", "start": 90, "end": 100},
]
UPLOAD_STAGE = "uploading results"
MAX_PROGRESS_BEFORE_DONE = 99.0     # 100 % only once the results are stored on the backend


def lease_until(seconds: int) -> datetime:
    return utcnow() + timedelta(seconds=seconds)


def requeue_expired(db: Session) -> int:
    """Running tasks whose lease ran out go back to pending (or fail if no attempts are left)."""
    now = utcnow()
    expired = db.scalars(select(Task).where(Task.status == TaskStatus.running.value, Task.lease_expires_at < now)).all()
    for t in expired:
        msg = f"worker {t.worker_id or '?'} stopped responding (attempt {t.attempts}/{t.max_attempts})"
        if t.attempts < t.max_attempts:
            t.status, t.error = TaskStatus.pending.value, msg + "; re-queued"
        else:
            t.status, t.error, t.finished_at = TaskStatus.failed.value, msg, now
        t.worker_id, t.lease_expires_at = None, None
    if expired:
        db.commit()
    return len(expired)


def claim_next(db: Session, worker_id: str, types: list[str] | None, lease_seconds: int) -> Task | None:
    """Atomically move the oldest pending task to running for this worker. Safe with several workers:
    the conditional UPDATE only succeeds for one of them."""
    requeue_expired(db)
    q = select(Task.id).where(Task.status == TaskStatus.pending.value).order_by(Task.created_at)
    if types:
        q = q.where(Task.type.in_(types))
    for task_id in db.scalars(q.limit(10)).all():
        now = utcnow()
        res = db.execute(update(Task).where(Task.id == task_id, Task.status == TaskStatus.pending.value).values(
            status=TaskStatus.running.value, worker_id=worker_id, lease_expires_at=lease_until(lease_seconds),
            attempts=Task.attempts + 1, started_at=func.coalesce(Task.started_at, now), error=None,
            stage=None, progress=0.0, updated_at=now))
        db.commit()
        if res.rowcount == 1:
            return db.get(Task, task_id)
    return None


def queue_position(db: Session, task: Task) -> int | None:
    """1 = next to be claimed. None unless pending."""
    if task.status != TaskStatus.pending.value:
        return None
    ahead = db.scalar(select(func.count()).select_from(Task).where(
        Task.status == TaskStatus.pending.value, Task.created_at < task.created_at))
    return int(ahead or 0) + 1


def pipeline_args(params: dict) -> list[str]:
    """CLI flags for ML_Work/run_pipeline.py. Only options the user set are passed, so the pipeline's own
    defaults (Person A's choice) apply otherwise."""
    flags = {"method": "--method", "train_mode": "--train-mode", "fps": "--fps", "long_side": "--long-side",
             "trainer": "--trainer", "max_frames": "--max-frames"}
    out: list[str] = []
    for key, flag in flags.items():
        if params.get(key) is not None:
            out += [flag, str(params[key])]
    return out
