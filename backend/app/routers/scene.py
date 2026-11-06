"""Scene delivery for the viewer (spec 4.4: GET /projects/{id}/scene returns the splat URL and metadata).

GET /projects/{id}/scene                      splat / ply / points URLs, cameras, cleaned metadata   (user JWT)
GET /projects/{id}/artifacts                  what is stored for the project                         (user JWT)
GET /projects/{id}/files/{path}               one stored file, HTTP Range supported   (user JWT  or  ?t=<signed>)
GET /worker/projects/{id}/files/{path}        same, for the worker (Week 2 rebuild needs frames + sparse)

Only files inside a registered artifact are served, addressed by their path in Person A's layout
(e.g. export/scene.splat). Server filesystem paths are never returned: A's json files contain absolute
/kaggle/working/... paths, so scene_meta is filtered through a whitelist.
"""
from __future__ import annotations

import json
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings
from ..deps import get_db, get_owned_project, get_settings, oauth2_scheme, require_worker
from ..models import Artifact, Project, Task, TaskStatus
from ..schemas import FileRef, SceneOut
from ..security import create_file_token, decode_access_token, decode_file_token
from ..storage import resolve_in_project

router = APIRouter(tags=["scene"])

# scene_meta.json keys that are safe and useful for the viewer (anything else may hold server paths)
META_KEYS = ("num_gaussians", "bbox_min", "bbox_max", "median_position", "trainer", "iters", "ply_mb", "splat_mb",
             "num_cameras", "heldout", "coordinate_note")
INFO_KEYS = ("num_frames", "num_images", "has_points_ply", "pose_method", "methods", "checkpoints", "num_gaussians",
             "num_cameras", "files", "dangling_links", "duration_s", "width", "height", "fps", "codec")
CAMERA_KEYS = ("id", "img_name", "width", "height", "position", "rotation", "fx", "fy", "cx", "cy")
UP_VECTOR = [0.0, -1.0, 0.0]          # COLMAP/OpenCV world is +y down (see ML_Work README section 5)
MEDIA_TYPES = {".splat": "application/octet-stream", ".ply": "application/octet-stream", ".json": "application/json",
               ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".npy": "application/octet-stream",
               ".bin": "application/octet-stream", ".txt": "text/plain", ".pt": "application/octet-stream",
               ".mp4": "video/mp4", ".md": "text/markdown"}


def _artifacts(db: Session, project_id: str, version: str) -> dict[str, Artifact]:
    rows = db.scalars(select(Artifact).where(Artifact.project_id == project_id, Artifact.version == version))
    return {a.kind: a for a in rows}


def _covering_artifact(db: Session, project_id: str, rel: str) -> Artifact | None:
    for a in db.scalars(select(Artifact).where(Artifact.project_id == project_id)):
        if rel == a.path or ((a.meta or {}).get("is_dir") and rel.startswith(a.path.rstrip("/") + "/")):
            return a
    return None


def _file_response(db: Session, settings: Settings, project_id: str, rel: str) -> FileResponse:
    rel = rel.replace("\\", "/").lstrip("/")
    if _covering_artifact(db, project_id, rel) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such artifact file")
    try:
        path = resolve_in_project(settings, project_id, rel)
    except ValueError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such artifact file")
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Artifact file is missing from storage")
    return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
                        filename=path.name, content_disposition_type="inline")


@router.get("/projects/{project_id}/scene", response_model=SceneOut)
def get_scene(version: str = Query("original", max_length=64), project: Project = Depends(get_owned_project),
              db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    arts = _artifacts(db, project.id, version)
    if not {"splat", "cameras", "scene_meta"} <= set(arts):
        active = db.scalar(select(Task).where(Task.project_id == project.id,
                                              Task.status.in_((TaskStatus.pending.value, TaskStatus.running.value))))
        msg = "Reconstruction still in progress" if active else "No reconstructed scene for this project yet"
        raise HTTPException(status.HTTP_404_NOT_FOUND, msg if version == "original" else f"No scene version '{version}'")

    token, exp = create_file_token(project.id, settings.jwt_secret, settings.file_url_ttl_minutes)

    def url(rel: str) -> str:
        return f"/projects/{project.id}/files/{quote(rel)}?t={token}"

    def ref(a: Artifact | None, rel: str | None = None) -> FileRef | None:
        if a is None:
            return None
        rel = rel or a.path
        p = resolve_in_project(settings, project.id, rel)
        if not p.is_file():
            return None
        return FileRef(url=url(rel), size_bytes=p.stat().st_size, sha256=a.sha256 if rel == a.path else None)

    splat = ref(arts["splat"])
    if splat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Splat file is missing from storage")
    try:
        cams = json.loads(resolve_in_project(settings, project.id, arts["cameras"].path).read_text(encoding="utf-8"))
        meta = json.loads(resolve_in_project(settings, project.id, arts["scene_meta"].path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Stored scene metadata is unreadable")

    scene_art = arts.get("scene")
    clean_meta = {k: meta[k] for k in META_KEYS if k in meta}
    if scene_art and (scene_art.meta or {}).get("pose_method"):
        clean_meta["pose_method"] = scene_art.meta["pose_method"]
    return SceneOut(project_id=project.id, version=version, task_id=arts["splat"].task_id, splat=splat,
                    ply=ref(arts.get("ply")), points=ref(scene_art, "scene/points.ply"),
                    cameras_url=url(arts["cameras"].path),
                    cameras=[{k: c[k] for k in CAMERA_KEYS if k in c} for c in cams],
                    meta=clean_meta, up_vector=UP_VECTOR, urls_expire_at=exp)


@router.get("/projects/{project_id}/artifacts")
def list_artifacts(project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    rows = db.scalars(select(Artifact).where(Artifact.project_id == project.id).order_by(Artifact.kind))
    return [{"kind": a.kind, "version": a.version, "path": a.path, "is_dir": bool((a.meta or {}).get("is_dir")),
             "size_bytes": a.size_bytes, "sha256": a.sha256, "task_id": a.task_id, "created_at": a.created_at,
             "info": {k: v for k, v in (a.meta or {}).items() if k in INFO_KEYS}}
            for a in rows]


@router.get("/projects/{project_id}/files/{file_path:path}")
def get_file(project_id: str, file_path: str, t: str | None = Query(default=None),
             bearer: str | None = Depends(oauth2_scheme), db: Session = Depends(get_db),
             settings: Settings = Depends(get_settings)):
    """Auth: a signed ?t= token from GET /scene (for viewers), or the owner's Bearer JWT."""
    allowed = False
    if t is not None:
        allowed = decode_file_token(t, settings.jwt_secret) == project_id
    elif bearer:
        uid = decode_access_token(bearer, settings.jwt_secret)
        p = db.get(Project, project_id)
        allowed = p is not None and uid is not None and p.owner_id == uid
    if not allowed:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED if not (t or bearer) else status.HTTP_404_NOT_FOUND,
                            "Not authenticated" if not (t or bearer) else "No such artifact file")
    return _file_response(db, settings, project_id, file_path)


@router.get("/worker/projects/{project_id}/files/{file_path:path}", tags=["worker"],
            dependencies=[Depends(require_worker)])
def worker_get_file(project_id: str, file_path: str, db: Session = Depends(get_db),
                    settings: Settings = Depends(get_settings)):
    return _file_response(db, settings, project_id, file_path)
