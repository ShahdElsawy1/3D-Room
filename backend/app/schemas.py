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
    # week 2 tasks (empty for "reconstruct"): what to download, which stage to run, what to upload
    edit_id: str | None = None
    stage: str | None = None       # w2_segment | w2_track | run_edit | w2_rebuild
    stage_args: list[str] = []
    inputs: list[dict] = []        # [{kind, version, path, is_dir, required}] -> GET /worker/projects/{pid}/artifacts/{kind}
    outputs: list[dict] = []       # [{kind, version, path, is_dir, required}] -> chunked upload


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
    version: str | None = Field(default=None, max_length=64,
                                description="optional check; the backend takes the version from the task")


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


# ---------------------------------------------------------------- week 2: object edits (frontend)
class Click(BaseModel):
    x: int = Field(ge=0, description="pixel column in the frame (frames are the Week 1 size, e.g. 960x540)")
    y: int = Field(ge=0, description="pixel row")
    positive: bool = Field(default=True, description="false = 'not this' click to refine the mask")


class EditCreate(BaseModel):
    ref_frame: str = Field(min_length=1, max_length=200, description="frame name from GET /projects/{id}/frames")
    clicks: list[Click] = Field(min_length=1, max_length=30)


class ClicksIn(BaseModel):
    clicks: list[Click] = Field(min_length=1, max_length=30)


class AddFrame(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    clicks: list[Click] = Field(default_factory=list, max_length=30,
                                description="optional hint; empty = let the segmentation stage find the object")


class FramesPatch(BaseModel):
    remove: list[str] = Field(default_factory=list, description="frames where the mask is wrong: not edited")
    restore: list[str] = Field(default_factory=list, description="undo an earlier remove")
    add: list[AddFrame] = Field(default_factory=list, description="fallback: frames the tracker missed")


class PreviewIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    seeds: list[int] | None = Field(default=None, max_length=8, description="explicit seeds; default num_seeds random")
    num_seeds: int = Field(default=3, ge=1, le=8)
    model: Literal["klein", "qwen-turbo", "qwen"] | None = Field(default=None, description="default: server setting")
    precision: Literal["bf16", "bf16-fp32", "fp8-bf16", "fp16"] | None = None


class AcceptIn(BaseModel):
    seed: int = Field(description="one of the candidates' seeds")
    with_previous: bool = Field(default=False, description="also condition on the previous edited frame")


class RebuildIn(BaseModel):
    mode: Literal["finetune", "scratch"] = Field(
        default="finetune", description="fine-tune the existing splat (faster) or re-train from scratch")


class TaskBrief(_Out):
    id: str
    type: str
    status: str
    stage: str | None
    progress: float
    error: str | None
    created_at: datetime


class EditOut(_Out):
    id: str
    project_id: str
    status: str = Field(description="created | segmenting | mask_ready | tracking | tracked | previewing | "
                                    "preview_ready | editing | edited | rebuilding | done | <type>_failed | <type>_cancelled")
    ref_frame: str
    clicks: list[dict]
    prompt: str | None
    chosen_seed: int | None
    removed_frames: list[str]
    current_task: TaskBrief | None = None
    tasks: list[TaskBrief] = []
    ref_mask_url: str | None = None
    scene_version: str | None = Field(default=None, description="pass as ?version= to GET /scene once rebuilt")
    created_at: datetime
    updated_at: datetime


class FrameOut(BaseModel):
    name: str
    index: int
    url: str
    width: int | None = None
    height: int | None = None
    camera_id: int | None = None


class EditFrameOut(BaseModel):
    name: str
    url: str
    mask_url: str | None
    present: bool = Field(description="the object is in this frame (tracker) and the user did not remove it")
    tracked_present: bool
    removed: bool
    source: str | None = None
    score: float | None = None


class CandidateOut(BaseModel):
    seed: int
    url: str
    crop_url: str | None = None
    seconds: float | None = None


class SceneVersion(_Out):
    version: str
    edit_id: str | None
    prompt: str | None
    created_at: datetime | None
