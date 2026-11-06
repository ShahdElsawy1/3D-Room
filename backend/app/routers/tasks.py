"""Tasks as seen by the user, plus the progress endpoint the pipeline posts to.

POST /projects/{id}/reconstruct   submit the "3D reconstruction" task                     (user)
GET  /projects/{id}/tasks         task history of a project                               (user)
GET  /tasks/{id}                  status, stage, progress 0-100, error                    (user)
POST /tasks/{id}/cancel           cancel a pending/running task                           (user)
POST /tasks/{id}/progress         real stage progress from ML_Work/common/progress.py     (worker token)
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings
from ..deps import get_current_user, get_db, get_owned_project, get_settings, require_worker
from ..models import Edit, Project, Task, TaskStatus, TaskType, User, utcnow
from ..schemas import ProgressAck, ProgressIn, ReconstructRequest, StageInfo, TaskOut
from ..tasks import (ACTIVE, FINAL, MAX_PROGRESS_BEFORE_DONE, RECONSTRUCT_STAGES, lease_until, queue_position,
                     requeue_expired)

router = APIRouter(tags=["tasks"])

STAGES_BY_TYPE = {TaskType.reconstruct.value: RECONSTRUCT_STAGES}


def task_out(db: Session, t: Task) -> TaskOut:
    out = TaskOut.model_validate(t)
    out.queue_position = queue_position(db, t)
    out.stages = [StageInfo(**s) for s in STAGES_BY_TYPE.get(t.type, [])]
    return out


def get_owned_task(task_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> Task:
    t = db.get(Task, task_id)
    if t is None or t.project.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
    return t


@router.post("/projects/{project_id}/reconstruct", response_model=TaskOut, status_code=status.HTTP_202_ACCEPTED)
def reconstruct(body: ReconstructRequest | None = None, project: Project = Depends(get_owned_project),
                db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    if not project.video_path:
        raise HTTPException(status.HTTP_409_CONFLICT, "Upload a video first")
    requeue_expired(db)
    if db.scalar(select(Task.id).where(Task.project_id == project.id, Task.status.in_(ACTIVE)).limit(1)):
        raise HTTPException(status.HTTP_409_CONFLICT, "This project already has a pending or running task")
    if db.scalar(select(Edit.id).where(Edit.project_id == project.id).limit(1)):
        # new frames would not match the edits' masks and edited frames
        raise HTTPException(status.HTTP_409_CONFLICT, "This project has edits; delete them before reconstructing again")
    params = (body or ReconstructRequest()).model_dump(exclude_none=True)
    t = Task(project_id=project.id, type=TaskType.reconstruct.value, params=params,
             max_attempts=settings.max_attempts, stage="queued")
    db.add(t)
    db.commit()
    return task_out(db, t)


@router.get("/projects/{project_id}/tasks", response_model=list[TaskOut])
def project_tasks(project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    requeue_expired(db)
    rows = db.scalars(select(Task).where(Task.project_id == project.id).order_by(Task.created_at.desc()))
    return [task_out(db, t) for t in rows]


@router.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(task: Task = Depends(get_owned_task), db: Session = Depends(get_db)):
    if requeue_expired(db):
        db.refresh(task)
    return task_out(db, task)


@router.post("/tasks/{task_id}/cancel", response_model=TaskOut)
def cancel_task(task: Task = Depends(get_owned_task), db: Session = Depends(get_db)):
    if task.status in FINAL:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Task is already {task.status}")
    task.status, task.finished_at = TaskStatus.cancelled.value, utcnow()
    task.error = "cancelled by user"
    db.commit()            # a running worker sees cancel=true on its next heartbeat/progress post and stops
    return task_out(db, task)


@router.post("/tasks/{task_id}/progress", response_model=ProgressAck, dependencies=[Depends(require_worker)])
def post_progress(task_id: str, body: ProgressIn, db: Session = Depends(get_db),
                  settings: Settings = Depends(get_settings)):
    """Body = ML_Work/common/progress.py record: {status, stage, progress, error, t}.

    - running: stage/progress update (renews the worker's lease)
    - failed:  the error is recorded; the WORKER then decides retry vs. final failure (POST /worker/tasks/{id}/fail)
    - done:    the pipeline finished; the task stays running at 99 % until the worker has uploaded the
               results and calls /worker/tasks/{id}/complete
    """
    t = db.get(Task, task_id)
    if t is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
    if t.status == TaskStatus.cancelled.value:
        return ProgressAck(ok=False, cancel=True)
    if t.status != TaskStatus.running.value:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Task is {t.status}, not running")
    t.stage = body.stage
    t.progress = min(round(body.progress, 1), MAX_PROGRESS_BEFORE_DONE)
    if body.status == "failed":
        t.error = (body.error or "stage failed")[:20000]
    t.lease_expires_at = lease_until(settings.lease_seconds)
    db.commit()
    return ProgressAck()
