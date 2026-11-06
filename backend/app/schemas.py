"""Request/response bodies. These are the contract for Person C's frontend (also visible at /docs)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_serializer, field_validator


class _Out(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    @field_serializer("*", when_used="json")
    def _utc(self, v):
        # DB stores naive UTC; emit ISO-8601 with an explicit Z so browsers parse it correctly
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        return v


# ---------------------------------------------------------------- auth
class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def _bcrypt_limit(cls, v: str) -> str:
        if len(v.encode("utf-8")) > 72:
            raise ValueError("password must be at most 72 bytes")
        return v


class UserOut(_Out):
    id: str
    email: EmailStr
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int          # seconds
    user: UserOut


# ---------------------------------------------------------------- projects
class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=5000)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)


class VideoMeta(BaseModel):
    filename: str
    size_bytes: int
    duration_s: float
    width: int
    height: int
    fps: float | None = None
    codec: str | None = None
    rotation: int = 0
    warnings: list[str] = []


class ProjectOut(_Out):
    id: str
    name: str
    description: str
    has_video: bool
    video: VideoMeta | None
    latest_task_id: str | None = None
    latest_task_status: str | None = None
    created_at: datetime
    updated_at: datetime


class VideoUploadOut(BaseModel):
    project_id: str
    video: VideoMeta


# ---------------------------------------------------------------- tasks
class ReconstructRequest(BaseModel):
    """All optional. Omitted fields use ML_Work/run_pipeline.py defaults (Person A's choices)."""
    method: Literal["vggt", "glomap", "both"] | None = Field(
        default=None, description="pose estimation; 'both' runs the VGGT-vs-GLOMAP comparison")
    train_mode: Literal["quick", "full"] | None = Field(default=None, description="quick ~7k iters, full ~30k")
    fps: Literal["auto", "1", "2", "3", "4"] | None = Field(default=None, description="frame sampling (spec: 1-4 fps)")
    long_side: int | None = Field(default=None, ge=640, le=960, description="frame long side in px (spec: 640-960)")
    trainer: Literal["gsplat", "inria"] | None = None
    max_frames: int | None = Field(default=None, ge=20, le=1000)


class StageInfo(BaseModel):
    name: str
    start: float
    end: float


class TaskOut(_Out):
    id: str
    project_id: str
    type: str
    status: str
    stage: str | None
    progress: float
    error: str | None
    params: dict
    result: dict | None
    attempts: int
    max_attempts: int
    queue_position: int | None = None
    stages: list[StageInfo] = []
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    updated_at: datetime


# ---------------------------------------------------------------- worker protocol
class ProgressIn(BaseModel):
    """Exactly what ML_Work/common/progress.py posts."""
    model_config = ConfigDict(extra="ignore")
    status: Literal["running", "done", "failed"] = "running"
    stage: str = Field(max_length=64)
    progress: float = Field(ge=0, le=100)
    error: str | None = None
    t: float | None = None


class ProgressAck(BaseModel):
    ok: bool = True
    cancel: bool = False


class ClaimIn(BaseModel):
    worker_id: str = Field(min_length=1, max_length=100)
    types: list[str] | None = Field(default=None, description="task types this worker can run; default all")


class ClaimOut(BaseModel):
    task: TaskOut
    lease_seconds: int
    video_url: str | None          # relative to the backend URL; needs the worker token
    video_filename: str | None
    video_sha256: str | None
    video_size_bytes: int | None
    pipeline_args: list[str]       # extra flags for run_pipeline.py


class WorkerIn(BaseModel):
    worker_id: str = Field(min_length=1, max_length=100)


class HeartbeatIn(WorkerIn):
    stage: str | None = Field(default=None, max_length=64)
    progress: float | None = Field(default=None, ge=0, le=100)


class HeartbeatOut(_Out):
    status: str
    cancel: bool
    lease_expires_at: datetime | None


class CompleteIn(WorkerIn):
    result: dict = Field(default_factory=dict)


class FailIn(WorkerIn):
    error: str = Field(max_length=20000)
    retryable: bool = True


# ---------------------------------------------------------------- artifact uploads (worker)
class UploadCreate(WorkerIn):
    kind: str = Field(description="one of the artifact kinds in app/artifacts.py (frames, scene, splat, ...)")
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_size: int | None = Field(default=None, description="bytes per chunk; default BACKEND_UPLOAD_CHUNK_MB")
    version: Literal["original"] = "original"


class UploadOut(BaseModel):
    upload_id: str
    kind: str
    path: str
    is_archive: bool
    size_bytes: int
    chunk_size: int
    total_chunks: int
    status: Literal["open", "complete"]
    received: int
    missing: list[int]
    meta: dict | None = None


class ChunkAck(BaseModel):
    index: int
    received: int
    total_chunks: int
    duplicate: bool


# ---------------------------------------------------------------- scene (frontend viewer)
class FileRef(BaseModel):
    url: str
    size_bytes: int
    sha256: str | None = None


class SceneOut(BaseModel):
    project_id: str
    version: str
    task_id: str | None
    splat: FileRef
    ply: FileRef | None = None
    points: FileRef | None = None          # coloured sparse cloud (scene/points.ply): three.js fallback
    cameras_url: str
    cameras: list[dict]
    meta: dict
    up_vector: list[float]
    urls_expire_at: datetime
