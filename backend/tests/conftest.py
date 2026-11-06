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


# ------------------------------------------------------------------ artifact helpers
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import tarfile  # noqa: E402

WORKER_ID = "test-worker"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tar_bytes(root: Path, top: str, symlinks: dict[str, str] | None = None, gz: bool = False) -> bytes:
    """tar of root/top with entries named top/..., plus optional extra symlink entries {name: target}."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz" if gz else "w") as tf:
        tf.add(root / top, arcname=top)
        for name, target in (symlinks or {}).items():
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            tf.addfile(info)
    return buf.getvalue()


def fake_pipeline_output(root: Path, n_frames: int = 3) -> Path:
    """Project-shaped folder in Person A's layout (paths/names as the pipeline writes them; contents minimal)."""
    (root / "frames" / "images").mkdir(parents=True)
    for i in range(1, n_frames + 1):
        (root / "frames" / "images" / f"{i:06d}.jpg").write_bytes(b"\xff\xd8\xff" + bytes([i]) * 100)
    (root / "frames" / "frames_index.json").write_text(json.dumps({"kept": n_frames, "frames": []}))
    sp = root / "scene" / "sparse" / "0"
    sp.mkdir(parents=True)
    for s in ("cameras", "images", "points3D"):
        (sp / f"{s}.txt").write_text(f"# {s}\n")
    (root / "scene" / "chosen.json").write_text(json.dumps({"method": "glomap", "from": "/kaggle/working/x"}))
    (root / "scene" / "points.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n" + b"\0" * 30)
    (root / "export").mkdir()
    (root / "export" / "scene.splat").write_bytes(b"\1" * 32 * 10)
    (root / "export" / "scene.ply").write_bytes(b"ply\n" + b"\0" * 1000)
    cams = [{"id": i, "img_name": f"{i:06d}.jpg", "width": 960, "height": 540, "position": [0, 0, i],
             "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "fx": 800.0, "fy": 800.0, "cx": 480.0, "cy": 270.0,
             "secret_path": "/kaggle/working/workspace/demo"} for i in range(1, n_frames + 1)]
    (root / "export" / "cameras.json").write_text(json.dumps(cams))
    (root / "export" / "scene_meta.json").write_text(json.dumps({
        "num_gaussians": 10, "splat_mb": 0.0, "trainer": "gsplat", "iters": 7000, "cameras": "cameras.json",
        "source": "converted from /kaggle/working/workspace/demo/gs/ckpt_6999.pt", "num_cameras": n_frames}))
    return root


def upload(client, worker, task_id, kind, data: bytes, chunk_size=4096, expect=200, worker_id=WORKER_ID):
    r = client.post(f"/worker/tasks/{task_id}/uploads", headers=worker,
                    json={"worker_id": worker_id, "kind": kind, "size_bytes": len(data), "sha256": sha(data),
                          "chunk_size": chunk_size})
    assert r.status_code == 200, r.text
    u = r.json()
    for i in u["missing"]:
        part = data[i * u["chunk_size"]:(i + 1) * u["chunk_size"]]
        c = client.put(f"/worker/uploads/{u['upload_id']}/chunks/{i}", content=part,
                       headers={**worker, "X-Worker-Id": worker_id, "X-Chunk-Sha256": sha(part)})
        assert c.status_code == 200, c.text
    r = client.post(f"/worker/uploads/{u['upload_id']}/complete", headers=worker, json={"worker_id": worker_id})
    assert r.status_code == expect, r.text
    return r.json()


def upload_reconstruction(client, worker, task_id, root: Path):
    """Upload in the order the worker uses: folders first (link targets exist), then export files."""
    upload(client, worker, task_id, "frames", tar_bytes(root, "frames"))
    upload(client, worker, task_id, "scene", tar_bytes(root, "scene", {"scene/images": "../frames/images"}))
    for kind, rel in (("splat", "export/scene.splat"), ("ply", "export/scene.ply"),
                      ("cameras", "export/cameras.json"), ("scene_meta", "export/scene_meta.json")):
        upload(client, worker, task_id, kind, (root / rel).read_bytes())
