"""Backend settings, read from environment variables (prefix BACKEND_).

Everything has a sensible local default so `uvicorn app.main:app` works with no setup.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"BACKEND_{name}", default)


@dataclass
class Settings:
    # Storage root. Each project gets <data_dir>/workspace/<project_id>/ with the SAME layout as
    # Person A's pipeline (ML_Work/common/config.py), so the worker and Week 2 read it unchanged.
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(BACKEND_DIR / "data"))))
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", ""))

    jwt_secret: str = field(default_factory=lambda: _env("JWT_SECRET", ""))
    jwt_expire_minutes: int = field(default_factory=lambda: int(_env("JWT_EXPIRE_MINUTES", str(7 * 24 * 60))))

    # Video upload rules (spec section 2: one room, 2-5 min). Outside the recommended range we only warn,
    # because short debug clips (30-60 s) are normal; outside the hard range we reject.
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "2048")))
    video_min_seconds: float = field(default_factory=lambda: float(_env("VIDEO_MIN_SECONDS", "5")))
    video_max_seconds: float = field(default_factory=lambda: float(_env("VIDEO_MAX_SECONDS", "600")))
    video_recommended_seconds: tuple[float, float] = (120.0, 300.0)
    ffprobe_bin: str = field(default_factory=lambda: _env("FFPROBE", "ffprobe"))

    # Pull-based worker (Kaggle). The token protects /worker/* and POST /tasks/{id}/progress.
    worker_token: str = field(default_factory=lambda: _env("WORKER_TOKEN", ""))
    # A running task whose worker sends nothing (progress or heartbeat) for this long is considered lost
    # and goes back to the queue (or fails once max_attempts is used up).
    lease_seconds: int = field(default_factory=lambda: int(_env("LEASE_SECONDS", "300")))
    max_attempts: int = field(default_factory=lambda: int(_env("MAX_ATTEMPTS", "2")))

    # Artifact uploads from the worker. Cloudflare limits one request to 100 MB, so chunks stay below that.
    upload_chunk_mb: int = field(default_factory=lambda: int(_env("UPLOAD_CHUNK_MB", "32")))
    upload_max_chunk_bytes: int = 90 * 1024 * 1024
    upload_min_chunk_bytes: int = 1024
    upload_max_artifact_mb: int = field(default_factory=lambda: int(_env("UPLOAD_MAX_ARTIFACT_MB", "20480")))
    # open uploads with no new chunk for this long are deleted (abandoned by a dead Kaggle session)
    upload_stale_hours: float = field(default_factory=lambda: float(_env("UPLOAD_STALE_HOURS", "24")))
    # lifetime of the signed file URLs returned by GET /projects/{id}/scene (viewers can't send auth headers)
    file_url_ttl_minutes: int = field(default_factory=lambda: int(_env("FILE_URL_TTL_MINUTES", "360")))

    cors_origins: list[str] = field(default_factory=lambda: [o.strip() for o in _env(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()])

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        if not self.database_url:
            self.database_url = f"sqlite:///{(self.data_dir / 'app.db').as_posix()}"
        # random secrets persisted in the data dir, so tokens survive restarts without any setup
        if not self.jwt_secret:
            self.jwt_secret = self._persisted_secret(".jwt_secret")
        if not self.worker_token:
            self.worker_token = self._persisted_secret(".worker_token")

    def _persisted_secret(self, name: str) -> str:
        p = self.data_dir / name
        if not p.exists():
            p.write_text(secrets.token_urlsafe(32))
        return p.read_text().strip()

    @property
    def workspace_dir(self) -> Path:
        return self.data_dir / "workspace"

    @property
    def tmp_dir(self) -> Path:
        d = self.data_dir / "tmp"
        d.mkdir(parents=True, exist_ok=True)
        return d
