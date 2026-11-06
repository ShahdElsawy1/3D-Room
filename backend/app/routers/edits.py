"""Week 2: click an object, type a prompt, see the edited 3D room (spec §4.5 UI flow). User JWT on every route.

GET    /projects/{id}/frames          training frames for the frame picker (signed image URLs, camera ids)
POST   /projects/{id}/edits           1. start an edit: reference frame + clicks -> 'segment' task (mask preview)
GET    /projects/{id}/edits           edit history of a project
GET    /edits/{eid}                   status, current task, reference-mask URL
DELETE /edits/{eid}                   remove an edit and its files (the original scene is untouched)
POST   /edits/{eid}/clicks            2. refine the mask with new positive/negative clicks -> 'segment'
POST   /edits/{eid}/track             4. propagate the mask to every frame -> 'track'
GET    /edits/{eid}/frames            4. grid: every frame with its mask URL and present/removed flags
PATCH  /edits/{eid}/frames            4. remove wrong frames / restore / add missed frames (fallback -> 'segment')
POST   /edits/{eid}/preview           3+5. prompt -> reference frame edited with several seeds -> 'edit_preview'
GET    /edits/{eid}/candidates        5. the candidates to choose from
POST   /edits/{eid}/accept            5. chosen seed -> every other frame -> 'edit_apply' (frame x of N)
POST   /edits/{eid}/rebuild           6. rebuild the splat from the edited frames, same poses -> 'rebuild'
GET    /projects/{id}/scene/versions  6. "original" + every rebuilt edit, for the before/after toggle
Progress of every step: GET /tasks/{task_id} as in Week 1 (edit.current_task.id).
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..artifacts import kind_spec
from ..config import Settings
from ..deps import get_current_user, get_db, get_owned_project, get_settings
from ..edits import edit_status, edit_tasks
from ..models import Artifact, Edit, Project, Task, User
from ..schemas import (AcceptIn, CandidateOut, ClicksIn, EditCreate, EditFrameOut, EditOut, FrameOut, FramesPatch,
                       PreviewIn, RebuildIn, SceneVersion, TaskBrief)
from ..security import create_file_token
from ..storage import project_dir, resolve_in_project
from ..tasks import ACTIVE, requeue_expired

router = APIRouter(tags=["edits (week 2)"])
IMAGE_EXT = (".jpg", ".jpeg", ".png")


# ------------------------------------------------------------------------------------------------- helpers
def get_owned_edit(edit_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> Edit:
    e = db.get(Edit, edit_id)
    if e is None or e.project.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Edit not found")
    return e


def _artifact(db: Session, project_id: str, kind: str, version: str) -> Artifact | None:
    return db.scalar(select(Artifact).where(Artifact.project_id == project_id, Artifact.kind == kind,
                                            Artifact.version == version))


def _require(db: Session, project_id: str, kind: str, version: str, msg: str) -> Artifact:
    a = _artifact(db, project_id, kind, version)
    if a is None:
        raise HTTPException(status.HTTP_409_CONFLICT, msg)
    return a


def _json(settings: Settings, project_id: str, rel: str):
    try:
        return json.loads(resolve_in_project(settings, project_id, rel).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _signer(settings: Settings, project_id: str):
    token, _ = create_file_token(project_id, settings.jwt_secret, settings.file_url_ttl_minutes)
    return lambda rel: f"/projects/{project_id}/files/{quote(rel)}?t={token}"


def _frames(db: Session, settings: Settings, project_id: str) -> list[str]:
    _require(db, project_id, "frames", "original", "No reconstructed frames yet: run the reconstruction first")
    d = project_dir(settings, project_id) / "frames" / "images"
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() in IMAGE_EXT) if d.is_dir() else []


def _cameras(db: Session, settings: Settings, project_id: str) -> dict[str, dict]:
    a = _artifact(db, project_id, "cameras", "original")
    cams = _json(settings, project_id, a.path) if a else None
    return {c["img_name"]: c for c in cams if isinstance(c, dict) and "img_name" in c} if isinstance(cams, list) else {}


def _check_clicks(clicks, frame: str, cams: dict[str, dict]) -> None:
    c = cams.get(frame)
    if c and c.get("width") and c.get("height"):
        bad = [k for k in clicks if k.x >= c["width"] or k.y >= c["height"]]
        if bad:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                                f"click ({bad[0].x},{bad[0].y}) is outside {frame} ({c['width']}x{c['height']})")


def _new_task(db: Session, settings: Settings, project: Project, task_type: str, params: dict) -> Task:
    """One task at a time per project (same rule as Week 1): the GPU worker runs them in order anyway."""
    requeue_expired(db)
    if db.scalar(select(Task.id).where(Task.project_id == project.id, Task.status.in_(ACTIVE)).limit(1)):
        raise HTTPException(status.HTTP_409_CONFLICT, "This project already has a pending or running task")
    t = Task(project_id=project.id, type=task_type, params=params, max_attempts=settings.max_attempts, stage="queued")
    db.add(t)
    return t


def _brief(t: Task | None) -> TaskBrief | None:
    return TaskBrief.model_validate(t) if t else None


def edit_out(db: Session, settings: Settings, e: Edit) -> EditOut:
    tasks = edit_tasks(db, e)
    sign = _signer(settings, e.project_id)
    ref_mask = None
    if _artifact(db, e.project_id, "masks", e.id):
        rel = f"{kind_spec('masks', e.id).path}/{Path(e.ref_frame).stem}.png"
        if resolve_in_project(settings, e.project_id, rel).is_file():
            ref_mask = sign(rel)
    return EditOut(id=e.id, project_id=e.project_id, status=edit_status(tasks), ref_frame=e.ref_frame,
                   clicks=e.clicks or [], prompt=e.prompt, chosen_seed=e.chosen_seed,
                   removed_frames=e.removed_frames or [], current_task=_brief(tasks[-1] if tasks else None),
                   tasks=[_brief(t) for t in tasks], ref_mask_url=ref_mask,
                   scene_version=e.id if _artifact(db, e.project_id, "splat", e.id) else None,
                   created_at=e.created_at, updated_at=e.updated_at)


def _segment_params(e: Edit) -> dict:
    return {"edit_id": e.id, "frame": e.ref_frame, "clicks": e.clicks}


def _edit_params(settings: Settings, e: Edit, model: str | None, precision: str | None) -> dict:
    precision = precision or settings.edit_precision
    if precision == "fp16" and not settings.edit_allow_fp16:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            "fp16 is not the spec's dtype; it needs BACKEND_EDIT_ALLOW_FP16=1 on the server")
    return {"edit_id": e.id, "prompt": e.prompt, "ref_frame": e.ref_frame, "model": model or settings.edit_model,
            "precision": precision, "placement": settings.edit_placement, "long_side": settings.edit_long_side,
            "allow_fp16": precision == "fp16"}


# ------------------------------------------------------------------------------------------------- frames
@router.get("/projects/{project_id}/frames", response_model=list[FrameOut])
def list_frames(project: Project = Depends(get_owned_project), db: Session = Depends(get_db),
                settings: Settings = Depends(get_settings)):
    names = _frames(db, settings, project.id)
    cams, sign = _cameras(db, settings, project.id), _signer(settings, project.id)
    return [FrameOut(name=n, index=i, url=sign(f"frames/images/{n}"), width=cams.get(n, {}).get("width"),
                     height=cams.get(n, {}).get("height"), camera_id=cams.get(n, {}).get("id"))
            for i, n in enumerate(names)]


# ------------------------------------------------------------------------------------------------- edits
@router.post("/projects/{project_id}/edits", response_model=EditOut, status_code=status.HTTP_201_CREATED)
def create_edit(body: EditCreate, project: Project = Depends(get_owned_project), db: Session = Depends(get_db),
                settings: Settings = Depends(get_settings)):
    if body.ref_frame not in _frames(db, settings, project.id):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"unknown frame {body.ref_frame!r}")
    _check_clicks(body.clicks, body.ref_frame, _cameras(db, settings, project.id))
    e = Edit(project_id=project.id, ref_frame=body.ref_frame, clicks=[c.model_dump() for c in body.clicks])
    db.add(e)
    db.flush()
    _new_task(db, settings, project, "segment", _segment_params(e))
    db.commit()
    return edit_out(db, settings, e)


@router.get("/projects/{project_id}/edits", response_model=list[EditOut])
def list_edits(project: Project = Depends(get_owned_project), db: Session = Depends(get_db),
               settings: Settings = Depends(get_settings)):
    requeue_expired(db)
    rows = db.scalars(select(Edit).where(Edit.project_id == project.id).order_by(Edit.created_at.desc()))
    return [edit_out(db, settings, e) for e in rows]


@router.get("/edits/{edit_id}", response_model=EditOut)
def get_edit(e: Edit = Depends(get_owned_edit), db: Session = Depends(get_db),
             settings: Settings = Depends(get_settings)):
    requeue_expired(db)
    return edit_out(db, settings, e)


@router.delete("/edits/{edit_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_edit(e: Edit = Depends(get_owned_edit), db: Session = Depends(get_db),
                settings: Settings = Depends(get_settings)):
    if any(t.status in ACTIVE for t in edit_tasks(db, e)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Cancel the edit's running task first")
    db.execute(delete(Artifact).where(Artifact.project_id == e.project_id, Artifact.version == e.id))
    shutil.rmtree(project_dir(settings, e.project_id) / "edits" / e.id, ignore_errors=True)
    db.delete(e)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/edits/{edit_id}/clicks", response_model=EditOut, status_code=status.HTTP_202_ACCEPTED)
def refine_clicks(body: ClicksIn, e: Edit = Depends(get_owned_edit), db: Session = Depends(get_db),
                  settings: Settings = Depends(get_settings)):
    _check_clicks(body.clicks, e.ref_frame, _cameras(db, settings, e.project_id))
    e.clicks = [c.model_dump() for c in body.clicks]
    _new_task(db, settings, e.project, "segment", _segment_params(e))
    db.commit()
    return edit_out(db, settings, e)


@router.post("/edits/{edit_id}/track", response_model=EditOut, status_code=status.HTTP_202_ACCEPTED)
def track(e: Edit = Depends(get_owned_edit), db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    _require(db, e.project_id, "masks", e.id, "No reference mask yet: wait for the segmentation")
    _new_task(db, settings, e.project, "track", {"edit_id": e.id})
    db.commit()
    return edit_out(db, settings, e)


@router.get("/edits/{edit_id}/frames", response_model=list[EditFrameOut])
def edit_frames(e: Edit = Depends(get_owned_edit), db: Session = Depends(get_db),
                settings: Settings = Depends(get_settings)):
    names = _frames(db, settings, e.project_id)
    a = _artifact(db, e.project_id, "masks", e.id)
    idx = (_json(settings, e.project_id, f"{a.path}/index.json") or {}) if a else {}
    by_name = {f.get("name"): f for f in idx.get("frames", []) if isinstance(f, dict)}
    removed, sign = set(e.removed_frames or []), _signer(settings, e.project_id)
    out = []
    for n in names:
        f = by_name.get(n, {})
        tracked = bool(f.get("present"))
        mask = f"{a.path}/{Path(n).stem}.png" if a and tracked else None
        out.append(EditFrameOut(name=n, url=sign(f"frames/images/{n}"), mask_url=sign(mask) if mask else None,
                                present=tracked and n not in removed, tracked_present=tracked, removed=n in removed,
                                source=f.get("source"), score=f.get("score")))
    return out


@router.patch("/edits/{edit_id}/frames", response_model=EditOut)
def patch_frames(body: FramesPatch, e: Edit = Depends(get_owned_edit), db: Session = Depends(get_db),
                 settings: Settings = Depends(get_settings)):
    names = set(_frames(db, settings, e.project_id))
    unknown = [n for n in [*body.remove, *body.restore, *(f.name for f in body.add)] if n not in names]
    if unknown:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"unknown frame(s): {unknown[:5]}")
    if e.ref_frame in body.remove:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "the reference frame cannot be removed")
    removed = (set(e.removed_frames or []) | set(body.remove)) - set(body.restore) - {f.name for f in body.add}
    e.removed_frames = sorted(removed)
    if body.add:
        _require(db, e.project_id, "masks", e.id, "No masks yet: segment the reference frame first")
        cams = _cameras(db, settings, e.project_id)
        for f in body.add:
            _check_clicks(f.clicks, f.name, cams)
        _new_task(db, settings, e.project, "segment",
                  {"edit_id": e.id, "add_frames": [{"name": f.name, "clicks": [c.model_dump() for c in f.clicks]}
                                                   for f in body.add]})
    db.commit()
    return edit_out(db, settings, e)


# ------------------------------------------------------------------------------------------------- diffusion
@router.post("/edits/{edit_id}/preview", response_model=EditOut, status_code=status.HTTP_202_ACCEPTED)
def preview(body: PreviewIn, e: Edit = Depends(get_owned_edit), db: Session = Depends(get_db),
            settings: Settings = Depends(get_settings)):
    _require(db, e.project_id, "masks", e.id, "No masks yet: segment (and track) the object first")
    e.prompt, e.chosen_seed = body.prompt.strip(), None
    seeds = body.seeds or random.sample(range(1, 2**31 - 1), body.num_seeds)
    params = {**_edit_params(settings, e, body.model, body.precision), "seeds": seeds}
    _new_task(db, settings, e.project, "edit_preview", params)
    db.commit()
    return edit_out(db, settings, e)


@router.get("/edits/{edit_id}/candidates", response_model=list[CandidateOut])
def candidates(e: Edit = Depends(get_owned_edit), db: Session = Depends(get_db),
               settings: Settings = Depends(get_settings)):
    a = _require(db, e.project_id, "candidates", e.id, "No candidates yet: run the preview first")
    idx = _json(settings, e.project_id, f"{a.path}/index.json") or {}
    sign, base = _signer(settings, e.project_id), a.path.rsplit("/", 1)[0]       # edits/<eid>
    return [CandidateOut(seed=s["seed"], url=sign(f"{base}/{s['file']}"),
                         crop_url=sign(f"{base}/{s['crop']}") if s.get("crop") else None, seconds=s.get("seconds"))
            for s in idx.get("seeds", [])]


@router.post("/edits/{edit_id}/accept", response_model=EditOut, status_code=status.HTTP_202_ACCEPTED)
def accept(body: AcceptIn, e: Edit = Depends(get_owned_edit), db: Session = Depends(get_db),
           settings: Settings = Depends(get_settings)):
    a = _require(db, e.project_id, "candidates", e.id, "No candidates yet: run the preview first")
    idx = _json(settings, e.project_id, f"{a.path}/index.json") or {}
    if body.seed not in [s.get("seed") for s in idx.get("seeds", [])]:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"seed {body.seed} is not one of the candidates")
    prev = next((t for t in reversed(edit_tasks(db, e)) if t.type == "edit_preview" and t.status == "done"), None)
    model, precision = (prev.params.get("model"), prev.params.get("precision")) if prev else (None, None)
    e.chosen_seed = body.seed
    params = {**_edit_params(settings, e, model, precision), "chosen_seed": body.seed,
              "exclude": e.removed_frames or [], "with_previous": body.with_previous}
    _new_task(db, settings, e.project, "edit_apply", params)
    db.commit()
    return edit_out(db, settings, e)


# ------------------------------------------------------------------------------------------------- rebuild
@router.post("/edits/{edit_id}/rebuild", response_model=EditOut, status_code=status.HTTP_202_ACCEPTED)
def rebuild(body: RebuildIn | None = None, e: Edit = Depends(get_owned_edit), db: Session = Depends(get_db),
            settings: Settings = Depends(get_settings)):
    _require(db, e.project_id, "edited", e.id, "No edited frames yet: accept a candidate first")
    _new_task(db, settings, e.project, "rebuild", {"edit_id": e.id, "mode": (body or RebuildIn()).mode})
    db.commit()
    return edit_out(db, settings, e)


@router.get("/projects/{project_id}/scene/versions", response_model=list[SceneVersion])
def scene_versions(project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    splats = db.scalars(select(Artifact).where(Artifact.project_id == project.id, Artifact.kind == "splat")).all()
    edits = {e.id: e for e in db.scalars(select(Edit).where(Edit.project_id == project.id))}
    out = []
    for a in sorted(splats, key=lambda a: (a.version != "original", a.created_at)):
        e = edits.get(a.version)
        out.append(SceneVersion(version=a.version, edit_id=e.id if e else None, prompt=e.prompt if e else None,
                                created_at=a.created_at))
    return out
