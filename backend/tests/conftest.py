import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# keep the module-level `app` in app.main from touching the real backend/data folder
os.environ.setdefault("BACKEND_DATA_DIR", tempfile.mkdtemp(prefix="backend_import_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", jwt_secret="test-secret-0123456789-abcdefghijklmnop", max_upload_mb=5,
                    video_min_seconds=2, video_max_seconds=60)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def signup(client, email="a@example.com", password="password123"):
    r = client.post("/auth/signup", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def auth(client):
    return signup(client)


def make_video(path: Path, seconds: float, size="640x360", rate=10) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size={size}:rate={rate}",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture(scope="session")
def video_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("videos")
