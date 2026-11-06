"""Pull-based worker API (spec 4.3). The Kaggle notebook only makes outgoing requests to these routes.
All need `Authorization: Bearer <BACKEND_WORKER_TOKEN>`.

GET  /worker/ping                        token check at worker start-up
POST /worker/claim                       -> 200 {task, video_url, pipeline_args, ...} | 204 nothing to do
POST /worker/tasks/{id}/heartbeat        renew the lease; reply says whether the user cancelled
GET  /worker/tasks/{id}/video            download the project's input video
POST /worker/tasks/{id}/complete         results stored -> done
POST /worker/tasks/{id}/fail             error -> retry (pending) or failed
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..config import Settings
from ..deps import get_db, get_settings, require_worker
from ..models import Task, TaskStatus, utcnow
from ..schemas import ClaimIn, ClaimOut, CompleteIn, FailIn, HeartbeatIn, HeartbeatOut
from ..storage import resolve_in_project
from ..tasks import MAX_PROGRESS_BEFORE_DONE, claim_next, lease_until, pipeline_args
from .tasks import task_out

router = APIRouter(prefix="/worker", tags=["worker"], dependencies=[Depends(require_worker)])


def _task_for_worker(db: Session, task_id: str, worker_id: str, allow_cancelled: bool = False) -> Task:
    t = db.get(Task, task_id)
    if t is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
    if t.status == TaskStatus.cancelled.value and allow_cancelled:
        return t
    if t.status != TaskStatus.running.value or t.worker_id != worker_id:
        # lease expired and the task was re-queued / taken by another worker, or it was cancelled
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"Task is {t.status} and not leased to worker '{worker_id}'; stop working on it")
    return t


@router.get("/ping")
def ping(settings: Settings = Depends(get_settings)):
    """Worker start-up check. ML_Work's progress.py ignores HTTP errors, so a wrong token would otherwise
    only show up as a progress bar that never moves."""
    return {"ok": True, "lease_seconds": settings.lease_seconds}


@router.post("/claim", response_model=ClaimOut, responses={204: {"description": "no pending task"}})
def claim(body: ClaimIn, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    t = claim_next(db, body.worker_id, body.types, settings.lease_seconds)
    if t is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    video = next((a for a in t.project.artifacts if a.kind == "video"), None)
    return ClaimOut(task=task_out(db, t), lease_seconds=settings.lease_seconds,
                    video_url=f"/worker/tasks/{t.id}/video" if t.project.video_path else None,
                    video_filename=t.project.video_path.split("/")[-1] if t.project.video_path else None,
                    video_sha256=video.sha256 if video else None,
                    video_size_bytes=video.size_bytes if video else None,
                    pipeline_args=pipeline_args(t.params or {}))


@router.post("/tasks/{task_id}/heartbeat", response_model=HeartbeatOut)
def heartbeat(task_id: str, body: HeartbeatIn, db: Session = Depends(get_db),
              settings: Settings = Depends(get_settings)):
    t = _task_for_worker(db, task_id, body.worker_id, allow_cancelled=True)
    if t.status == TaskStatus.running.value:
        t.lease_expires_at = lease_until(settings.lease_seconds)
        if body.stage is not None:
            t.stage = body.stage
        if body.progress is not None:
            t.progress = min(round(body.progress, 1), MAX_PROGRESS_BEFORE_DONE)
        db.commit()
    return HeartbeatOut(status=t.status, cancel=t.status == TaskStatus.cancelled.value,
                        lease_expires_at=t.lease_expires_at)


@router.get("/tasks/{task_id}/video")
def task_video(task_id: str, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    t = db.get(Task, task_id)
    if t is None or not t.project.video_path:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No video for this task")
    path = resolve_in_project(settings, t.project_id, t.project.video_path)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video file missing from storage")
    return FileResponse(path, filename=path.name)


@router.post("/tasks/{task_id}/complete")
def complete(task_id: str, body: CompleteIn, db: Session = Depends(get_db)):
    t = _task_for_worker(db, task_id, body.worker_id)
    t.status, t.progress, t.stage = TaskStatus.done.value, 100.0, "done"
    t.error, t.result, t.finished_at, t.lease_expires_at = None, body.result, utcnow(), None
    db.commit()
    return {"ok": True, "status": t.status}


@router.post("/tasks/{task_id}/fail")
def fail(task_id: str, body: FailIn, db: Session = Depends(get_db)):
    t = _task_for_worker(db, task_id, body.worker_id)
    t.error = body.error[:20000]
    t.worker_id, t.lease_expires_at = None, None
    if body.retryable and t.attempts < t.max_attempts:
        t.status, t.stage, t.progress = TaskStatus.pending.value, "queued (retry)", 0.0
    else:
        t.status, t.finished_at = TaskStatus.failed.value, utcnow()
    db.commit()
    return {"ok": True, "status": t.status, "attempts": t.attempts, "max_attempts": t.max_attempts}
