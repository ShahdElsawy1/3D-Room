"""Resumable chunked artifact upload (worker token). Needed because Cloudflare caps a request at 100 MB and
scene.ply / checkpoints are larger, and because a Kaggle network hiccup must not restart a 300 MB upload.

POST /worker/tasks/{task_id}/uploads          create, or RESUME an identical earlier upload (same kind + sha256)
GET  /worker/uploads/{upload_id}              which chunks the server already has
PUT  /worker/uploads/{upload_id}/chunks/{i}   raw bytes of chunk i (any order; repeats are harmless)
                                              optional header X-Chunk-Sha256 -> chunk rejected if it doesn't match
POST /worker/uploads/{upload_id}/complete     verify whole-file sha256, validate, stage for the task

Folder artifacts (frames/, scene/, ...) are uploaded as ONE tar (optionally gzip) whose entries all start with
that folder's name, e.g. frames/images/000001.jpg. Header for every chunk PUT: X-Worker-Id.
"""
from __future__ import annotations

import hashlib
import math
import shutil
import tarfile

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..artifacts import KINDS, ArtifactError, extract_archive, parts_dir, staging_dir, validate
from ..config import Settings
from ..deps import get_db, get_settings, require_worker
from ..models import Task, TaskStatus, UploadChunk, UploadSession
from ..schemas import ChunkAck, UploadCreate, UploadOut, WorkerIn
from ..storage import project_dir, sha256_file
from ..tasks import lease_until

router = APIRouter(prefix="/worker", tags=["worker uploads"], dependencies=[Depends(require_worker)])


def _leased_task(db: Session, task_id: str, worker_id: str, settings: Settings) -> Task:
    t = db.get(Task, task_id)
    if t is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
    if t.status != TaskStatus.running.value or t.worker_id != worker_id:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"Task is {t.status} and not leased to worker '{worker_id}'; stop uploading")
    t.lease_expires_at = lease_until(settings.lease_seconds)     # uploads keep the lease alive
    return t


def _session(db: Session, upload_id: str) -> UploadSession:
    s = db.get(UploadSession, upload_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Upload not found")
    return s


def _part(settings: Settings, s: UploadSession):
    return parts_dir(settings) / f"{s.id}.part"


def _out(db: Session, s: UploadSession) -> UploadOut:
    have = set(db.scalars(select(UploadChunk.index).where(UploadChunk.upload_id == s.id)))
    if s.status == "complete":
        have = set(range(s.total_chunks))
    return UploadOut(upload_id=s.id, kind=s.kind, path=s.path, is_archive=s.is_archive, size_bytes=s.size_bytes,
                     chunk_size=s.chunk_size, total_chunks=s.total_chunks, status=s.status, received=len(have),
                     missing=[i for i in range(s.total_chunks) if i not in have], meta=s.meta)


@router.post("/tasks/{task_id}/uploads", response_model=UploadOut)
def create_upload(task_id: str, body: UploadCreate, db: Session = Depends(get_db),
                  settings: Settings = Depends(get_settings)):
    task = _leased_task(db, task_id, body.worker_id, settings)
    if body.kind not in KINDS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            f"Unknown artifact kind '{body.kind}'. Allowed: {', '.join(KINDS)}")
    if body.size_bytes > settings.upload_max_artifact_mb * 2**20:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Artifact larger than BACKEND_UPLOAD_MAX_ARTIFACT_MB")

    # Already done: this exact content is what is currently staged for this kind -> nothing to send.
    staged = db.scalar(select(UploadSession).where(
        UploadSession.task_id == task.id, UploadSession.kind == body.kind, UploadSession.status == "complete")
        .order_by(UploadSession.updated_at.desc()))
    if staged is not None and staged.sha256 == body.sha256 and staged.size_bytes == body.size_bytes:
        db.commit()
        return _out(db, staged)
    # Resume: an unfinished upload of the same content -> hand it back with the list of missing chunks.
    resumable = db.scalar(select(UploadSession).where(
        UploadSession.task_id == task.id, UploadSession.kind == body.kind, UploadSession.status == "open",
        UploadSession.sha256 == body.sha256, UploadSession.size_bytes == body.size_bytes)
        .order_by(UploadSession.created_at.desc()))
    if resumable is not None and _part(settings, resumable).exists():
        db.commit()
        return _out(db, resumable)

    chunk = body.chunk_size or settings.upload_chunk_mb * 2**20
    if not settings.upload_min_chunk_bytes <= chunk <= settings.upload_max_chunk_bytes:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            f"chunk_size must be {settings.upload_min_chunk_bytes}-{settings.upload_max_chunk_bytes} bytes")
    k = KINDS[body.kind]
    s = UploadSession(project_id=task.project_id, task_id=task.id, kind=body.kind, version=body.version,
                      path=k.path, is_archive=k.is_dir, size_bytes=body.size_bytes, sha256=body.sha256,
                      chunk_size=chunk, total_chunks=math.ceil(body.size_bytes / chunk))
    db.add(s)
    db.flush()
    with _part(settings, s).open("wb") as f:          # preallocate: chunks are written at their offset
        f.truncate(body.size_bytes)
    db.commit()
    return _out(db, s)


@router.get("/uploads/{upload_id}", response_model=UploadOut)
def upload_status(upload_id: str, db: Session = Depends(get_db)):
    return _out(db, _session(db, upload_id))


@router.put("/uploads/{upload_id}/chunks/{index}", response_model=ChunkAck)
async def put_chunk(upload_id: str, index: int, request: Request, x_worker_id: str = Header(...),
                    x_chunk_sha256: str | None = Header(default=None), db: Session = Depends(get_db),
                    settings: Settings = Depends(get_settings)):
    s = _session(db, upload_id)
    if s.status == "complete":
        return ChunkAck(index=index, received=s.total_chunks, total_chunks=s.total_chunks, duplicate=True)
    _leased_task(db, s.task_id, x_worker_id, settings)
    if not 0 <= index < s.total_chunks:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"chunk index must be 0-{s.total_chunks - 1}")
    expected = s.chunk_size if index < s.total_chunks - 1 else s.size_bytes - s.chunk_size * (s.total_chunks - 1)
    part = _part(settings, s)
    if not part.exists():
        raise HTTPException(status.HTTP_409_CONFLICT, "Upload staging file is gone; create the upload again")

    h, n = hashlib.sha256(), 0
    buf = bytearray()
    async for piece in request.stream():
        n += len(piece)
        if n > expected:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"chunk {index} must be {expected} bytes")
        h.update(piece)
        buf += piece
    if n != expected:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            f"chunk {index} is {n} bytes, expected {expected} (interrupted? send it again)")
    digest = h.hexdigest()
    if x_chunk_sha256 and x_chunk_sha256.lower() != digest:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"chunk {index} checksum mismatch; send it again")
    with part.open("r+b") as f:                       # idempotent: a repeated chunk rewrites the same bytes
        f.seek(index * s.chunk_size)
        f.write(buf)

    row = db.get(UploadChunk, (s.id, index))
    duplicate = row is not None
    if row is None:
        db.add(UploadChunk(upload_id=s.id, index=index, size_bytes=n, sha256=digest))
    else:
        row.size_bytes, row.sha256 = n, digest
    db.commit()
    received = len(db.scalars(select(UploadChunk.index).where(UploadChunk.upload_id == s.id)).all())
    return ChunkAck(index=index, received=received, total_chunks=s.total_chunks, duplicate=duplicate)


@router.post("/uploads/{upload_id}/complete", response_model=UploadOut)
def complete_upload(upload_id: str, body: WorkerIn, db: Session = Depends(get_db),
                    settings: Settings = Depends(get_settings)):
    s = _session(db, upload_id)
    if s.status == "complete":
        return _out(db, s)
    _leased_task(db, s.task_id, body.worker_id, settings)
    out = _out(db, s)
    if out.missing:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            {"message": f"{len(out.missing)} chunk(s) missing", "missing": out.missing[:1000]})
    part = _part(settings, s)
    actual = sha256_file(part)
    if actual != s.sha256:
        # corrupted somewhere: forget every chunk so the worker re-sends the file
        db.execute(delete(UploadChunk).where(UploadChunk.upload_id == s.id))
        db.commit()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            f"sha256 mismatch (expected {s.sha256[:12]}..., got {actual[:12]}...); "
                            "all chunks were discarded, upload the file again")

    stage = staging_dir(settings, s.task_id)
    tmp = stage.parent / f".tmp_{s.id}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        if s.is_archive:
            info = extract_archive(part, s.path, tmp, [stage, project_dir(settings, s.project_id)],
                                   max_bytes=settings.upload_max_artifact_mb * 2**20)
        else:
            (tmp / s.path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(part, tmp / s.path)
            info = {}
        info.update(validate(s.kind, tmp))
    except (ArtifactError, tarfile.TarError, OSError, EOFError) as e:
        shutil.rmtree(tmp, ignore_errors=True)
        db.execute(delete(UploadChunk).where(UploadChunk.upload_id == s.id))
        db.commit()
        detail = str(e) if isinstance(e, ArtifactError) else f"not a valid archive: {e}"
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"{s.kind} rejected: {detail}")

    dst = stage / s.path                              # staged; moved into the project when the task completes
    if dst.exists():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp / s.path), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    part.unlink(missing_ok=True)
    s.status, s.meta = "complete", info
    db.execute(delete(UploadChunk).where(UploadChunk.upload_id == s.id))
    db.commit()
    return _out(db, s)
