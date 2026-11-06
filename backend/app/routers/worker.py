"""Pull-based worker API (spec 4.3). The Kaggle notebook only makes outgoing requests to these routes.
All need `Authorization: Bearer <BACKEND_WORKER_TOKEN>`.

GET  /worker/ping                        token check at worker start-up
POST /worker/claim                       -> 200 {task, video_url, pipeline_args, ...} | 204 nothing to do
POST /worker/tasks/{id}/heartbeat        renew the lease; reply says whether the user cancelled
GET  /worker/tasks/{id}/video            download the project's input video
GET  /worker/projects/{pid}/artifacts/{kind}?version=   one stored artifact (folders as a .tar), week 2 inputs
POST /worker/tasks/{id}/complete         results stored -> done
POST /worker/tasks/{id}/fail             error -> retry (pending) or failed
"""
from __future__ import annotations

import tarfile
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from ..artifacts import ArtifactError, missing_required, promote
from ..edits import STAGE, inputs_for, is_edit_task, outputs_for, stage_args
from ..models import Artifact
from ..config import Settings
from ..deps import get_db, get_settings, require_worker
from ..maintenance import maybe_cleanup
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
def claim(body: ClaimIn, request: Request, db: Session = Depends(get_db),
          settings: Settings = Depends(get_settings)):
    maybe_cleanup(request.app.state, db, settings)
    t = claim_next(db, body.worker_id, body.types, settings.lease_seconds)
    if t is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    video = next((a for a in t.project.artifacts if a.kind == "video"), None)
    w2 = {}
    if is_edit_task(t.type):
        w2 = dict(edit_id=t.params["edit_id"], stage=STAGE[t.type], stage_args=stage_args(t),
                  inputs=inputs_for(t), outputs=outputs_for(t))
    return ClaimOut(task=task_out(db, t), lease_seconds=settings.lease_seconds, **w2,
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


@router.get("/projects/{project_id}/artifacts/{kind}")
def artifact_download(project_id: str, kind: str, version: str = Query("original", max_length=64),
                      db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    """A whole stored artifact for a week 2 stage: a file as is, a folder as an uncompressed .tar whose entries
    are named by their path in the project (e.g. frames/images/000001.jpg), to be extracted into the project."""
    a = db.scalar(select(Artifact).where(Artifact.project_id == project_id, Artifact.kind == kind,
                                         Artifact.version == version))
    if a is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No {kind} artifact (version {version})")
    path = resolve_in_project(settings, project_id, a.path)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Artifact missing from storage")
    if path.is_file():
        return FileResponse(path, filename=path.name)
    out = settings.tmp_dir / f"{uuid.uuid4().hex}.tar"
    with tarfile.open(out, "w") as tf:
        tf.add(path, arcname=a.path)
    return FileResponse(out, filename=f"{kind}.tar", media_type="application/x-tar",
                        background=BackgroundTask(out.unlink, missing_ok=True))


@router.post("/tasks/{task_id}/complete")
def complete(task_id: str, body: CompleteIn, db: Session = Depends(get_db),
             settings: Settings = Depends(get_settings)):
    """Only succeeds once every required artifact of this task is uploaded and verified; then all of the task's
    artifacts are moved into the project folder together (no mix of old and new results)."""
    t = _task_for_worker(db, task_id, body.worker_id)
    missing = missing_required(db, t.type, t.id)
    if missing:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            {"message": "required artifacts not uploaded", "missing": missing})
    try:
        stored = promote(db, settings, t)
    except (ArtifactError, OSError) as e:
        db.rollback()
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"could not store artifacts: {e}")
    t.status, t.progress, t.stage = TaskStatus.done.value, 100.0, "done"
    t.error, t.result, t.finished_at, t.lease_expires_at = None, body.result, utcnow(), None
    db.commit()
    return {"ok": True, "status": t.status, "artifacts": sorted(a.kind for a in stored)}


@router.post("/tasks/{task_id}/fail")
def fail(task_id: str, body: FailIn, db: Session = Depends(get_db)):
    t = _task_for_worker(db, task_id, body.worker_id)
    # keep the precise error the failing stage reported via /progress (e.g. "CUDA out of memory") as the
    # headline, followed by the worker's details (log tail)
    stage_error = t.error
    t.error = (f"{stage_error}\n\n{body.error}" if stage_error and stage_error not in body.error
               else body.error)[:20000]
    t.worker_id, t.lease_expires_at = None, None
    if body.retryable and t.attempts < t.max_attempts:
        t.status, t.stage, t.progress = TaskStatus.pending.value, "queued (retry)", 0.0
    else:
        t.status, t.finished_at = TaskStatus.failed.value, utcnow()
    db.commit()
    return {"ok": True, "status": t.status, "attempts": t.attempts, "max_attempts": t.max_attempts}
