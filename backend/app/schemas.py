"""Request/response bodies. These are the contract for Person C's frontend (also visible at /docs)."""
from __future__ import annotations

from datetime import datetime, timezone

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
