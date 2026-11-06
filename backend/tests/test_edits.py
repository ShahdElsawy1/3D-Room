"""M8: Week 2 edit API (frames, edits, segment/track/preview/accept/rebuild tasks, versions) and the worker loop."""
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("requests")
from PIL import Image  # noqa: E402

from app.config import Settings  # noqa: E402
from app.models import Project  # noqa: E402
from conftest import WORKER_ID, fake_pipeline_output, tar_bytes, upload, upload_reconstruction  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]
WORKER = BACKEND / "worker" / "kaggle_worker.py"
STUBS = BACKEND / "tests" / "stubs_w2"
N = 5


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", jwt_secret="test-secret-0123456789-abcdefghijklmnop",
                    edit_model="fake")


def real_output(root: Path, n=N) -> Path:
    """Person A's layout with decodable 960x540 frames (an object that moves right)."""
    fake_pipeline_output(root, n)
    rng = np.random.default_rng(0)
    for i in range(1, n + 1):
        img = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
        img[200:360, 300 + 20 * i: 520 + 20 * i] = (150, 60, 40)
        Image.fromarray(img).save(root / "frames" / "images" / f"{i:06d}.jpg", quality=92)
    return root


def reconstructed(client, settings, auth, tmp_path):
    pid = client.post("/projects", json={"name": "Room"}, headers=auth).json()["id"]
    with client.app.state.SessionLocal() as db:            # skip the video upload: not what this file tests
        db.get(Project, pid).video_path = "video/input.mp4"
        db.commit()
    tid = client.post(f"/projects/{pid}/reconstruct", headers=auth, json={}).json()["id"]
    W = {"Authorization": f"Bearer {settings.worker_token}"}
    assert client.post("/worker/claim", headers=W, json={"worker_id": WORKER_ID}).status_code == 200
    upload_reconstruction(client, W, tid, real_output(tmp_path / "out"))
    r = client.post(f"/worker/tasks/{tid}/complete", headers=W, json={"worker_id": WORKER_ID, "result": {}})
    assert r.status_code == 200, r.text
    return pid, W


def claim(client, W, types=None):
    r = client.post("/worker/claim", headers=W, json={"worker_id": WORKER_ID, "types": types})
    assert r.status_code == 200, r.text
    return r.json()


def finish(client, W, tid):
    r = client.post(f"/worker/tasks/{tid}/complete", headers=W, json={"worker_id": WORKER_ID, "result": {}})
    assert r.status_code == 200, r.text


def masks_tar(tmp: Path, eid: str, present: dict[str, bool]) -> bytes:
    d = tmp / "m" / "edits" / eid / "masks"
    d.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, p in present.items():
        if p:
            m = np.zeros((540, 960), np.uint8)
            m[200:360, 320:540] = 255
            Image.fromarray(m).save(d / (Path(name).stem + ".png"))
        frames.append({"name": name, "present": p, "source": "click", "score": 0.9, "area_px": 1})
    (d / "index.json").write_text(json.dumps({"ref_frame": "000001.jpg", "method": "t", "frames": frames}))
    return tar_bytes(tmp / "m", f"edits/{eid}/masks")


# ---------------------------------------------------------------------------------------------------- API only
def test_frames_list_and_edit_preconditions(client, settings, auth, tmp_path):
    pid0 = client.post("/projects", json={"name": "empty"}, headers=auth).json()["id"]
    assert client.get(f"/projects/{pid0}/frames", headers=auth).status_code == 409   # nothing reconstructed
    pid, W = reconstructed(client, settings, auth, tmp_path)
    frames = client.get(f"/projects/{pid}/frames", headers=auth).json()
    assert [f["name"] for f in frames] == [f"{i:06d}.jpg" for i in range(1, N + 1)]
    assert frames[0]["width"] == 960 and frames[0]["camera_id"] == 1
    img = client.get(frames[0]["url"])                              # signed URL, no auth header
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"

    r = client.post(f"/projects/{pid}/edits", headers=auth, json={"ref_frame": "nope.jpg", "clicks": [{"x": 1, "y": 1}]})
    assert r.status_code == 422
    r = client.post(f"/projects/{pid}/edits", headers=auth, json={"ref_frame": "000001.jpg", "clicks": [{"x": 960, "y": 5}]})
    assert r.status_code == 422 and "outside" in r.text
    r = client.post(f"/projects/{pid}/edits", headers=auth,
                    json={"ref_frame": "000001.jpg", "clicks": [{"x": 400, "y": 280}, {"x": 10, "y": 10, "positive": False}]})
    assert r.status_code == 201, r.text
    e = r.json()
    assert e["status"] == "segmenting" and e["current_task"]["type"] == "segment"
    eid = e["id"]
    # one task at a time per project; later steps need the mask
    assert client.post(f"/edits/{eid}/clicks", headers=auth, json={"clicks": [{"x": 5, "y": 5}]}).status_code == 409
    assert client.post(f"/edits/{eid}/preview", headers=auth, json={"prompt": "x"}).status_code == 409
    assert client.post(f"/edits/{eid}/accept", headers=auth, json={"seed": 1}).status_code == 409
    assert client.post(f"/edits/{eid}/rebuild", headers=auth, json={}).status_code == 409
    # reconstructing again would invalidate the edit
    assert client.post(f"/projects/{pid}/reconstruct", headers=auth, json={}).status_code == 409
    # other users can't see it
    other = client.post("/auth/signup", json={"email": "b@example.com", "password": "password123"}).json()
    assert client.get(f"/edits/{eid}", headers={"Authorization": f"Bearer {other['access_token']}"}).status_code == 404

    c = claim(client, W)
    assert c["stage"] == "w2_segment" and c["edit_id"] == eid
    assert c["stage_args"][:4] == ["--edit-id", eid, "--frame", "000001.jpg"]
    assert json.loads(c["stage_args"][5])[1] == {"x": 10, "y": 10, "positive": False}
    assert {i["kind"]: (i["version"], i["path"], i["required"]) for i in c["inputs"]} == {
        "frames": ("original", "frames", True), "cameras": ("original", "export/cameras.json", False),
        "masks": (eid, f"edits/{eid}/masks", False)}
    assert c["outputs"] == [{"kind": "masks", "version": eid, "path": f"edits/{eid}/masks", "is_dir": True,
                             "required": True}]
    tid = c["task"]["id"]
    # the worker can fetch inputs; a missing optional one is a 404
    fr = client.get(f"/worker/projects/{pid}/artifacts/frames", headers=W)
    assert fr.status_code == 200 and fr.headers["content-type"] == "application/x-tar"
    assert client.get(f"/worker/projects/{pid}/artifacts/masks?version={eid}", headers=W).status_code == 404
    assert client.get(f"/worker/projects/{pid}/artifacts/frames", headers=auth).status_code == 401
    # an edit task can't write the original scene, and only edit kinds are allowed
    r = client.post(f"/worker/tasks/{tid}/uploads", headers=W, json={"worker_id": WORKER_ID, "kind": "masks",
                    "size_bytes": 10, "sha256": "0" * 64, "version": "original"})
    assert r.status_code == 422 and "version" in r.text
    r = client.post(f"/worker/tasks/{tid}/uploads", headers=W, json={"worker_id": WORKER_ID, "kind": "frames",
                    "size_bytes": 10, "sha256": "0" * 64})
    assert r.status_code == 422
    # masks without a png for a present frame are rejected
    bad = masks_tar(tmp_path / "bad", eid, {"000001.jpg": True})
    import tarfile, io  # noqa: E401
    buf = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(bad)) as src, tarfile.open(fileobj=buf, mode="w") as dst:
        for m in src.getmembers():
            if not m.name.endswith(".png"):
                dst.addfile(m, src.extractfile(m) if m.isfile() else None)
    upload(client, W, tid, "masks", buf.getvalue(), expect=422)
    assert client.post(f"/worker/tasks/{tid}/complete", headers=W,
                       json={"worker_id": WORKER_ID, "result": {}}).status_code == 409   # masks required
    upload(client, W, tid, "masks", masks_tar(tmp_path, eid, {f"{i:06d}.jpg": i < N for i in range(1, N + 1)}))
    finish(client, W, tid)

    e = client.get(f"/edits/{eid}", headers=auth).json()
    assert e["status"] == "mask_ready" and e["ref_mask_url"]
    m = client.get(e["ref_mask_url"])
    assert m.status_code == 200 and m.headers["content-type"] == "image/png"
    arts = {(a["kind"], a["version"]): a["path"] for a in client.get(f"/projects/{pid}/artifacts", headers=auth).json()}
    assert arts[("masks", eid)] == f"edits/{eid}/masks" and ("frames", "original") in arts

    grid = client.get(f"/edits/{eid}/frames", headers=auth).json()
    assert [g["present"] for g in grid] == [True, True, True, True, False]
    assert grid[-1]["mask_url"] is None and grid[0]["mask_url"]
    assert client.patch(f"/edits/{eid}/frames", headers=auth, json={"remove": ["000001.jpg"]}).status_code == 422
    r = client.patch(f"/edits/{eid}/frames", headers=auth, json={"remove": ["000003.jpg", "000004.jpg"]})
    assert r.json()["removed_frames"] == ["000003.jpg", "000004.jpg"]
    r = client.patch(f"/edits/{eid}/frames", headers=auth, json={"restore": ["000004.jpg"]})
    assert r.json()["removed_frames"] == ["000003.jpg"] and r.json()["status"] == "mask_ready"   # no task
    grid = client.get(f"/edits/{eid}/frames", headers=auth).json()
    assert grid[2]["removed"] and not grid[2]["present"] and grid[2]["tracked_present"]

    # fp16 is refused unless the server allows it
    r = client.post(f"/edits/{eid}/preview", headers=auth, json={"prompt": "green sofa", "precision": "fp16"})
    assert r.status_code == 422 and "BACKEND_EDIT_ALLOW_FP16" in r.text
    r = client.post(f"/edits/{eid}/preview", headers=auth, json={"prompt": " green sofa ", "seeds": [7, 8]})
    assert r.status_code == 202 and r.json()["prompt"] == "green sofa"
    c = claim(client, W)
    assert c["stage"] == "run_edit" and c["stage_args"][0] == "preview"
    a = c["stage_args"]
    assert a[a.index("--seeds") + 1] == "7,8" and a[a.index("--model") + 1] == "fake"
    assert a[a.index("--precision") + 1] == "bf16" and "--allow-fp16" not in a

    # deleting needs the task stopped; then everything of the edit is gone, the original stays
    assert client.delete(f"/edits/{eid}", headers=auth).status_code == 409
    assert client.post(f"/tasks/{c['task']['id']}/cancel", headers=auth).status_code == 200
    assert client.delete(f"/edits/{eid}", headers=auth).status_code == 204
    assert not (settings.workspace_dir / pid / "edits" / eid).exists()
    kinds = {(a["kind"], a["version"]) for a in client.get(f"/projects/{pid}/artifacts", headers=auth).json()}
    assert all(v == "original" for _, v in kinds) and ("splat", "original") in kinds
    assert client.post(f"/projects/{pid}/reconstruct", headers=auth, json={}).status_code == 202


# ---------------------------------------------------------------------------------------------------- live worker
@pytest.fixture
def server(client):
    import uvicorn
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(client.app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(timeout=10)


def run_worker(base, settings, root, extra_env=None):
    cmd = [sys.executable, str(WORKER), "--backend-url", base, "--token", settings.worker_token, "--root", str(root),
           "--worker-id", "w2-worker", "--types", "segment,track,edit_preview,edit_apply,rebuild",
           "--stages-dir", str(STUBS), "--once", "--heartbeat-seconds", "2", "--poll-seconds", "1"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(extra_env or {}))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    return p.stdout


def wait_status(client, auth, eid, want):
    e = client.get(f"/edits/{eid}", headers=auth).json()
    assert e["status"] == want, json.dumps(e, indent=1)
    return e


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_full_week2_flow_through_the_worker(client, settings, auth, tmp_path, server):
    pid, W = reconstructed(client, settings, auth, tmp_path)
    proj = settings.workspace_dir / pid
    original = {p.relative_to(proj).as_posix(): sha(p) for p in proj.rglob("*") if p.is_file()}
    root = tmp_path / "kaggle"

    # 1. click -> mask
    eid = client.post(f"/projects/{pid}/edits", headers=auth,
                      json={"ref_frame": "000002.jpg", "clicks": [{"x": 450, "y": 280}]}).json()["id"]
    log = run_worker(server, settings, root)
    assert "inputs downloaded: frames, cameras" in log
    e = wait_status(client, auth, eid, "mask_ready")
    assert client.get(e["ref_mask_url"]).status_code == 200
    # 2. refine
    client.post(f"/edits/{eid}/clicks", headers=auth, json={"clicks": [{"x": 460, "y": 270}]})
    run_worker(server, settings, root)
    wait_status(client, auth, eid, "mask_ready")
    # 4. track: the stand-in loses the object in the last frame
    client.post(f"/edits/{eid}/track", headers=auth)
    assert "inputs downloaded: frames, scene, cameras, masks" in run_worker(server, settings, root)
    wait_status(client, auth, eid, "tracked")
    grid = client.get(f"/edits/{eid}/frames", headers=auth).json()
    assert [g["present"] for g in grid] == [True, True, True, True, False]
    # 4. fallback: mark the missed frame by hand, remove a wrong one
    client.patch(f"/edits/{eid}/frames", headers=auth,
                 json={"add": [{"name": "000005.jpg", "clicks": [{"x": 560, "y": 280}]}], "remove": ["000004.jpg"]})
    run_worker(server, settings, root)
    wait_status(client, auth, eid, "mask_ready")
    grid = {g["name"]: g for g in client.get(f"/edits/{eid}/frames", headers=auth).json()}
    assert grid["000005.jpg"]["present"] and grid["000005.jpg"]["source"] == "user"
    assert grid["000004.jpg"]["removed"] and grid["000001.jpg"]["present"]        # earlier masks kept

    # 3+5. prompt -> candidates (Person B's real stage with the fake model)
    client.post(f"/edits/{eid}/preview", headers=auth, json={"prompt": "dark green leather sofa", "seeds": [11, 12]})
    run_worker(server, settings, root)
    e = wait_status(client, auth, eid, "preview_ready")
    t = client.get(f"/tasks/{e['current_task']['id']}", headers=auth).json()
    assert t["result"]["edit"]["model"] == "fake" and t["result"]["uploads"]["candidates"]["mb"] > 0
    cands = client.get(f"/edits/{eid}/candidates", headers=auth).json()
    assert [c["seed"] for c in cands] == [11, 12]
    assert client.get(cands[1]["url"]).headers["content-type"] == "image/png"
    # 5. accept -> every other present frame
    client.post(f"/edits/{eid}/accept", headers=auth, json={"seed": 12})
    run_worker(server, settings, root)
    e = wait_status(client, auth, eid, "edited")
    assert e["chosen_seed"] == 12
    t = client.get(f"/tasks/{e['current_task']['id']}", headers=auth).json()
    s = t["result"]["edit"]["summary"]
    assert s["frames"] == 3 and s["degenerate_frames"] == 0                       # 1, 3, 5 (2 = ref, 4 removed)
    edited = json.loads((proj / "edits" / eid / "edited" / "index.json").read_text())
    assert sorted(x["frame"] for x in edited["edited"]) == ["000001.jpg", "000002.jpg", "000003.jpg", "000005.jpg"]
    assert edited["excluded_by_user"] == ["000004.jpg"]
    for x in edited["edited"]:                                                     # same size as the originals
        assert Image.open(proj / "edits" / eid / x["file"]).size == (960, 540)

    # 6. rebuild -> a second scene version; the original is untouched
    client.post(f"/edits/{eid}/rebuild", headers=auth, json={"mode": "finetune"})
    log = run_worker(server, settings, root)
    assert "edited" in log and "scene" in log
    e = wait_status(client, auth, eid, "done")
    assert e["scene_version"] == eid
    t = client.get(f"/tasks/{e['current_task']['id']}", headers=auth).json()
    assert t["result"]["stages"]["s4_rebuild"]["edited_frames"] == 4            # numbers only (A strips paths)
    after = client.get(f"/projects/{pid}/scene?version={eid}", headers=auth).json()
    before = client.get(f"/projects/{pid}/scene", headers=auth).json()
    assert after["version"] == eid and after["splat"]["size_bytes"] == 32 * 7
    assert before["version"] == "original" and before["splat"]["size_bytes"] == 32 * 10
    assert client.get(after["splat"]["url"]).content == b"\2" * 32 * 7
    versions = client.get(f"/projects/{pid}/scene/versions", headers=auth).json()
    assert [(v["version"], v["prompt"]) for v in versions] == [("original", None), (eid, "dark green leather sofa")]
    now = {p.relative_to(proj).as_posix(): sha(p) for p in proj.rglob("*") if p.is_file()}
    assert {k: now[k] for k in original} == original                               # nothing original changed
    assert all(k.startswith(f"edits/{eid}/") for k in set(now) - set(original))


def test_missing_stage_and_stage_failure(client, settings, auth, tmp_path, server):
    pid, W = reconstructed(client, settings, auth, tmp_path)
    eid = client.post(f"/projects/{pid}/edits", headers=auth,
                      json={"ref_frame": "000001.jpg", "clicks": [{"x": 450, "y": 280}]}).json()["id"]
    # a failing stage is retried, then reported with its own error
    run_worker(server, settings, tmp_path / "k", {"STUB_W2_FAIL": "segment"})
    e = wait_status(client, auth, eid, "segmenting")                               # re-queued (attempt 1 of 2)
    run_worker(server, settings, tmp_path / "k", {"STUB_W2_FAIL": "segment"})
    e = wait_status(client, auth, eid, "segment_failed")
    t = client.get(f"/tasks/{e['current_task']['id']}", headers=auth).json()
    assert "SAM out of memory" in t["error"] and t["attempts"] == 2
    # a worker without Person A's stage scripts fails the task clearly and does not retry it
    client.post(f"/edits/{eid}/clicks", headers=auth, json={"clicks": [{"x": 450, "y": 280}]})
    cmd = [sys.executable, str(WORKER), "--backend-url", server, "--token", settings.worker_token, "--root",
           str(tmp_path / "k2"), "--types", "segment", "--stages-dir", str(tmp_path / "nothing"), "--once"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert p.returncode == 0, p.stderr
    e = wait_status(client, auth, eid, "segment_failed")
    t = client.get(f"/tasks/{e['current_task']['id']}", headers=auth).json()
    assert "w2_segment.py" in t["error"] and t["attempts"] == 1
