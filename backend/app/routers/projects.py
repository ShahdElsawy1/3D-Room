"""Project management + video upload.

POST   /projects                 create
GET    /projects                 list the current user's projects (never global)
GET    /projects/{id}            one project
PATCH  /projects/{id}            rename / describe
DELETE /projects/{id}            delete project, its tasks and all stored files (refused while a task is active)
POST   /projects/{id}/video      upload the room video (multipart field "file")
GET    /projects/{id}/video      stream the stored video back (supports HTTP Range for <video> seeking)
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings
from ..deps import get_current_user, get_db, get_owned_project, get_settings
from ..models import Artifact, Project, Task, TaskStatus, User
from ..schemas import ProjectCreate, ProjectOut, ProjectUpdate, VideoMeta, VideoUploadOut
from ..storage import delete_project_dir, project_dir, resolve_in_project, sha256_file
from ..video import ALLOWED_EXTENSIONS, VideoError, check_rules, probe_video

router = APIRouter(prefix="/projects", tags=["projects"])

ACTIVE = (TaskStatus.pending.value, TaskStatus.running.value)
CHUNK = 1 << 20


def project_out(p: Project) -> ProjectOut:
    latest = p.tasks[-1] if p.tasks else None
    return ProjectOut(id=p.id, name=p.name, description=p.description, has_video=bool(p.video_path),
                      video=VideoMeta(**p.video_meta) if p.video_meta else None,
                      latest_task_id=latest.id if latest else None,
                      latest_task_status=latest.status if latest else None,
                      created_at=p.created_at, updated_at=p.updated_at)


def has_active_task(db: Session, project_id: str) -> bool:
    return db.scalar(select(Task.id).where(Task.project_id == project_id, Task.status.in_(ACTIVE)).limit(1)) is not None


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db),
                   settings: Settings = Depends(get_settings)):
    p = Project(owner_id=user.id, name=body.name.strip(), description=body.description)
    db.add(p)
    db.commit()
    project_dir(settings, p.id).mkdir(parents=True, exist_ok=True)
    return project_out(p)


@router.get("", response_model=list[ProjectOut])
def list_projects(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(Project).where(Project.owner_id == user.id).order_by(Project.created_at.desc()))
    return [project_out(p) for p in rows]


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project: Project = Depends(get_owned_project)):
    return project_out(project)


@router.patch("/{project_id}", response_model=ProjectOut)
def update_project(body: ProjectUpdate, project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    if body.name is not None:
        project.name = body.name.strip()
    if body.description is not None:
        project.description = body.description
    db.commit()
    return project_out(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project: Project = Depends(get_owned_project), db: Session = Depends(get_db),
                   settings: Settings = Depends(get_settings)):
    if has_active_task(db, project.id):
        raise HTTPException(status.HTTP_409_CONFLICT, "Project has a pending/running task; cancel it first")
    pid = project.id
    db.delete(project)
    db.commit()
    delete_project_dir(settings, pid)


@router.post("/{project_id}/video", response_model=VideoUploadOut, status_code=status.HTTP_201_CREATED)
def upload_video(request: Request, file: UploadFile = File(..., description="room video, 2-5 min recommended"),
                 project: Project = Depends(get_owned_project), db: Session = Depends(get_db),
                 settings: Settings = Depends(get_settings)):
    max_bytes = settings.max_upload_mb * 1024 * 1024
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes + 1024 * 1024:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"File larger than {settings.max_upload_mb} MB")
    if has_active_task(db, project.id):
        raise HTTPException(status.HTTP_409_CONFLICT, "A task is running on this project; the video cannot change now")

    filename = Path(file.filename or "video").name
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            f"Unsupported extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    # stream to a temp file (never hold the video in memory), enforcing the size limit while copying
    tmp = settings.tmp_dir / f"upload_{project.id}_{os.getpid()}{ext}"
    size = 0
    try:
        with tmp.open("wb") as out:
            while chunk := file.file.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"File larger than {settings.max_upload_mb} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")
        try:
            meta = probe_video(tmp, settings.ffprobe_bin)
            warnings = check_rules(meta, settings.video_min_seconds, settings.video_max_seconds,
                                   settings.video_recommended_seconds)
        except VideoError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Invalid video: {e}")

        # replace any previous upload (input.* only; derived artifacts are re-created by the next reconstruction)
        vdir = project_dir(settings, project.id) / "video"
        vdir.mkdir(parents=True, exist_ok=True)
        for old in vdir.glob("input.*"):
            old.unlink()
        rel = f"video/input{ext}"
        final = resolve_in_project(settings, project.id, rel)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()

    vmeta = VideoMeta(filename=re.sub(r"[^\w.\- ]", "_", filename), size_bytes=size, warnings=warnings, **meta)
    project.video_path = rel
    project.video_meta = vmeta.model_dump()
    for a in db.scalars(select(Artifact).where(Artifact.project_id == project.id, Artifact.kind == "video")):
        db.delete(a)
    db.add(Artifact(project_id=project.id, kind="video", path=rel, size_bytes=size,
                    sha256=sha256_file(final), meta=vmeta.model_dump()))
    db.commit()
    return VideoUploadOut(project_id=project.id, video=vmeta)


@router.get("/{project_id}/video")
def get_video(project: Project = Depends(get_owned_project), settings: Settings = Depends(get_settings)):
    if not project.video_path:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No video uploaded")
    path = resolve_in_project(settings, project.id, project.video_path)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video file missing from storage")
    return FileResponse(path, filename=(project.video_meta or {}).get("filename") or path.name)
