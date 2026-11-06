"""End-to-end worker tests: real uvicorn server (thread) + real worker process + CPU stub pipeline that runs
Person A's real CPU stages. Skipped when ffmpeg or A's Python deps (requirements-dev.txt) are missing."""
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest
from conftest import HAS_FFMPEG, make_video

pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("requests")
if not HAS_FFMPEG:
    pytest.skip("ffmpeg not on PATH", allow_module_level=True)

import requests  # noqa: E402
import uvicorn  # noqa: E402

from app.main import create_app  # noqa: E402
from app.models import Task, utcnow  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]
ML_DIR = BACKEND.parent / "ML_Work"
WORKER = BACKEND / "worker" / "kaggle_worker.py"
STUB = BACKEND / "tests" / "stub_pipeline.py"


@pytest.fixture
def server(settings):
    app = create_app(settings)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield app, f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(timeout=10)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("v") / "room.mp4", 6, size="1280x720", rate=10)


class Api:
    def __init__(self, base):
        self.base = base
        r = requests.post(f"{base}/auth/signup", json={"email": "w@example.com", "password": "password123"})
        self.h = {"Authorization": f"Bearer {r.json()['access_token']}"}

    def get(self, p):
        return requests.get(self.base + p, headers=self.h)

    def post(self, p, **kw):
        return requests.post(self.base + p, headers=self.h, **kw)


def submit(api, video, params=None):
    pid = api.post("/projects", json={"name": "Room"}).json()["id"]
    with open(video, "rb") as f:
        assert api.post(f"/projects/{pid}/video", files={"file": ("room.mp4", f)}).status_code == 201
    tid = api.post(f"/projects/{pid}/reconstruct", json=params or {"fps": "2"}).json()["id"]
    return pid, tid


def worker_cmd(base, token, root, *extra):
    return [sys.executable, str(WORKER), "--backend-url", base, "--token", token, "--pipeline", str(STUB),
            "--root", str(root), "--worker-id", "pytest-worker", "--heartbeat-seconds", "2", "--poll-seconds", "1",
            *extra]


def env(**kw):
    base = {"STUB_ML_DIR": str(ML_DIR), "STUB_TRAIN_SECONDS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    return {**os.environ, **base, **{k: str(v) for k, v in kw.items()}}


def run_worker(base, token, root, *extra, timeout=240, **envkw):
    return subprocess.run(worker_cmd(base, token, root, *extra), env=env(**envkw), capture_output=True, text=True,
                          timeout=timeout)


def alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for(pred, timeout=120, every=0.5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(every)
    raise AssertionError("condition not reached")


# ------------------------------------------------------------------ tests
def test_happy_path_end_to_end(server, settings, video, tmp_path):
    app, base = server
    api = Api(base)
    pid, tid = submit(api, video)
    assert api.get(f"/tasks/{tid}").json()["status"] == "pending"

    r = run_worker(base, settings.worker_token, tmp_path / "ws", "--once")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    t = api.get(f"/tasks/{tid}").json()
    assert (t["status"], t["progress"], t["error"]) == ("done", 100.0, None)

    res = t["result"]
    assert set(res["stages"]) >= {"s1_extract_frames", "s4_train_3dgs", "s5_export_splat"}
    assert res["stages"]["s1_extract_frames"]["frames_kept"] == 12           # 6 s at 2 fps (A's real stage 1)
    assert set(res["uploads"]) == {"frames", "splat", "cameras", "scene_meta", "scene", "poses", "metrics", "manifest"}
    assert "/kaggle" not in str(res) and str(tmp_path) not in str(res)          # numbers only, no paths

    scene = api.get(f"/projects/{pid}/scene").json()
    assert scene["meta"]["num_gaussians"] == 2000 and scene["meta"]["pose_method"] == "glomap"
    assert len(scene["cameras"]) == 12 and scene["ply"] is None                  # ply optional, off by default
    assert requests.get(base + scene["splat"]["url"]).content[:4]                # served
    kinds = {a["kind"] for a in api.get(f"/projects/{pid}/artifacts").json()}
    assert kinds == {"video", "frames", "splat", "cameras", "scene_meta", "scene", "poses", "metrics", "manifest"}

    P = settings.workspace_dir / pid                      # A's layout on the backend; images are real files
    assert (P / "scene/images/000001.jpg").is_file() and not (P / "scene/images").is_symlink()
    assert (P / "frames/images/000012.jpg").is_file() and (P / "poses/glomap/sparse/0/cameras.bin").is_file()
    assert not (P / "poses/glomap/images").exists()                              # duplicate of frames, skipped
    assert not (tmp_path / "ws" / pid).exists()                                  # worker cleaned its workspace
    assert "connected to" in r.stdout and "uploaded splat" in r.stdout
    # real stage progress reached the backend through A's progress.py with the worker token
    assert "401" not in r.stdout


def test_optional_ply_and_checkpoint(server, settings, video, tmp_path):
    app, base = server
    api = Api(base)
    pid, tid = submit(api, video)
    r = run_worker(base, settings.worker_token, tmp_path / "ws", "--once", "--upload-ply", "--upload-checkpoint")
    assert r.returncode == 0, r.stdout[-3000:]
    arts = {a["kind"]: a for a in api.get(f"/projects/{pid}/artifacts").json()}
    assert "ply" in arts and arts["gs"]["info"]["checkpoints"] == ["gsplat_7000/ckpts/ckpt_6999.pt"]
    assert api.get(f"/projects/{pid}/scene").json()["ply"]["size_bytes"] > 0
    # ply goes last: the viewer's splat is uploaded before it
    assert r.stdout.index("uploaded splat") < r.stdout.index("uploaded ply")


def test_pipeline_failure_is_retried_then_fails(server, settings, video, tmp_path):
    app, base = server
    api = Api(base)
    pid, tid = submit(api, video)
    r = run_worker(base, settings.worker_token, tmp_path / "ws", "--once", STUB_FAIL_AT="training")
    assert r.returncode == 0
    t = api.get(f"/tasks/{tid}").json()
    assert t["status"] == "pending" and t["stage"] == "queued (retry)" and "simulated failure" in t["error"]

    run_worker(base, settings.worker_token, tmp_path / "ws", "--once", STUB_FAIL_AT="training")
    t = api.get(f"/tasks/{tid}").json()
    assert t["status"] == "failed" and t["attempts"] == 2
    assert t["error"].startswith("stub: CUDA out of memory")         # the stage's own message is the headline
    assert "simulated failure in training" in t["error"] and "[progress]" not in t["error"]
    assert api.get(f"/projects/{pid}/scene").status_code == 404
    assert not (settings.data_dir / "staging" / tid).exists() or not any((settings.data_dir / "staging" / tid).rglob("*"))


def _run_until_training(base, settings, tmp_path, pidfile):
    proc = subprocess.Popen(worker_cmd(base, settings.worker_token, tmp_path / "ws", "--once"),
                            env=env(STUB_TRAIN_SECONDS=120, STUB_CHILD_PIDFILE=pidfile),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wait_for(lambda: pidfile.exists() and pidfile.read_text().strip(), timeout=180)
    return proc, int(pidfile.read_text())


def test_cancel_stops_pipeline_and_children(server, settings, video, tmp_path):
    app, base = server
    api = Api(base)
    pid, tid = submit(api, video)
    proc, child = _run_until_training(base, settings, tmp_path, tmp_path / "child.pid")
    assert alive(child)
    wait_for(lambda: api.get(f"/tasks/{tid}").json()["stage"] == "training", timeout=30)

    assert api.post(f"/tasks/{tid}/cancel").json()["status"] == "cancelled"
    out, _ = proc.communicate(timeout=60)                 # worker notices within one heartbeat (2 s)
    assert proc.returncode == 0, out[-2000:]
    assert "cancelled by user" in out
    wait_for(lambda: not alive(child), timeout=20)        # the GPU-holding grandchild is gone too
    t = api.get(f"/tasks/{tid}").json()
    assert t["status"] == "cancelled"
    assert api.get(f"/projects/{pid}/artifacts").json()[0]["kind"] == "video"   # nothing else stored
    assert len(api.get(f"/projects/{pid}/artifacts").json()) == 1


def test_lease_lost_mid_run_abandons_task(server, settings, video, tmp_path):
    app, base = server
    api = Api(base)
    pid, tid = submit(api, video)
    proc, child = _run_until_training(base, settings, tmp_path, tmp_path / "child.pid")
    # the backend decides this worker is gone (e.g. long network outage) and gives the task to someone else
    db = app.state.SessionLocal()
    db.get(Task, tid).lease_expires_at = utcnow() - timedelta(seconds=1)
    db.commit()
    db.close()
    other = requests.post(f"{base}/worker/claim", json={"worker_id": "other-worker"},
                          headers={"Authorization": f"Bearer {settings.worker_token}"}).json()
    assert other["task"]["id"] == tid and other["task"]["attempts"] == 2

    out, _ = proc.communicate(timeout=60)
    assert "lease lost" in out
    wait_for(lambda: not alive(child), timeout=20)
    t = api.get(f"/tasks/{tid}").json()
    assert t["status"] == "running"                       # still running - for the other worker, untouched


def test_bad_token_fails_fast(server, video, tmp_path):
    app, base = server
    r = run_worker(base, "wrong-token", tmp_path / "ws", "--once", timeout=60)
    assert r.returncode != 0 and "token rejected" in (r.stdout + r.stderr)


def test_no_task_and_idle_exit(server, settings, tmp_path):
    app, base = server
    r = run_worker(base, settings.worker_token, tmp_path / "ws", "--once")
    assert r.returncode == 0 and "no pending task" in r.stdout
    t0 = time.time()
    r = run_worker(base, settings.worker_token, tmp_path / "ws", "--max-idle-minutes", "0.05")
    assert r.returncode == 0 and "idle for" in r.stdout and time.time() - t0 < 30


def test_video_download_resumes(server, settings, video, tmp_path):
    """Unit-level: a partially downloaded video is completed with a Range request and verified."""
    sys.path.insert(0, str(BACKEND / "worker"))
    import kaggle_worker as kw
    app, base = server
    api = Api(base)
    pid, tid = submit(api, video)
    claim = requests.post(f"{base}/worker/claim", json={"worker_id": "dl"},
                          headers={"Authorization": f"Bearer {settings.worker_token}"}).json()
    data = Path(video).read_bytes()
    dest = tmp_path / "partial.mp4"
    dest.write_bytes(data[:1000])
    kw.download(kw.Api(base, settings.worker_token), claim["video_url"], dest, claim["video_size_bytes"],
                claim["video_sha256"])
    assert dest.read_bytes() == data
    dest.write_bytes(b"corrupt" + data[7:])                # same size, wrong content -> re-downloaded
    kw.download(kw.Api(base, settings.worker_token), claim["video_url"], dest, claim["video_size_bytes"],
                claim["video_sha256"])
    assert dest.read_bytes() == data
