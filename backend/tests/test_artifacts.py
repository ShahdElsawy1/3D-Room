import hashlib
import json
import os
import tarfile
from datetime import timedelta

import pytest
from conftest import (WORKER_ID, fake_pipeline_output, sha, signup, tar_bytes, upload, upload_reconstruction)

from app.models import Project, Task, utcnow
from app.security import create_file_token


@pytest.fixture
def worker(settings):
    return {"Authorization": f"Bearer {settings.worker_token}"}


@pytest.fixture
def running_task(client, auth, worker, settings):
    """A reconstruct task claimed by WORKER_ID (project video registered directly)."""
    pid = client.post("/projects", json={"name": "Room"}, headers=auth).json()["id"]
    db = client.app.state.SessionLocal()
    db.get(Project, pid).video_path = "video/input.mp4"
    db.commit()
    db.close()
    tid = client.post(f"/projects/{pid}/reconstruct", headers=auth).json()["id"]
    assert client.post("/worker/claim", json={"worker_id": WORKER_ID}, headers=worker).status_code == 200
    return pid, tid


@pytest.fixture
def out(tmp_path):
    return fake_pipeline_output(tmp_path / "pipeline_out")


def new_upload(client, worker, tid, kind, data, chunk_size=4096, claimed_sha=None):
    r = client.post(f"/worker/tasks/{tid}/uploads", headers=worker,
                    json={"worker_id": WORKER_ID, "kind": kind, "size_bytes": len(data),
                          "sha256": claimed_sha or sha(data), "chunk_size": chunk_size})
    assert r.status_code == 200, r.text
    return r.json()


def put(client, worker, uid, i, data, **headers):
    return client.put(f"/worker/uploads/{uid}/chunks/{i}", content=data,
                      headers={**worker, "X-Worker-Id": WORKER_ID, **headers})


def finish(client, worker, uid):
    return client.post(f"/worker/uploads/{uid}/complete", headers=worker, json={"worker_id": WORKER_ID})


def complete_task(client, worker, tid):
    return client.post(f"/worker/tasks/{tid}/complete", headers=worker, json={"worker_id": WORKER_ID})


# ------------------------------------------------------------------ happy path + scene
def test_full_reconstruction_upload_and_scene(client, auth, worker, settings, running_task, out):
    pid, tid = running_task
    assert client.get(f"/projects/{pid}/scene", headers=auth).json()["detail"] == "Reconstruction still in progress"
    upload_reconstruction(client, worker, tid, out)
    # staged only: nothing visible until the task completes
    assert client.get(f"/projects/{pid}/scene", headers=auth).status_code == 404
    assert not (settings.workspace_dir / pid / "export").exists()

    r = complete_task(client, worker, tid)
    assert r.status_code == 200 and r.json()["artifacts"] == ["cameras", "frames", "ply", "scene", "scene_meta", "splat"]

    # A's exact layout and names in the project folder; the scene/images symlink became a real copy
    P = settings.workspace_dir / pid
    for rel in ("frames/images/000001.jpg", "frames/frames_index.json", "scene/sparse/0/cameras.txt",
                "scene/sparse/0/images.txt", "scene/sparse/0/points3D.txt", "scene/chosen.json", "scene/points.ply",
                "scene/images/000003.jpg", "export/scene.splat", "export/scene.ply", "export/cameras.json",
                "export/scene_meta.json"):
        assert (P / rel).is_file(), rel
    assert not (P / "scene" / "images").is_symlink()
    assert (P / "export/scene.splat").read_bytes() == (out / "export/scene.splat").read_bytes()
    assert not (settings.data_dir / "staging" / tid).exists()

    scene = client.get(f"/projects/{pid}/scene", headers=auth)
    assert scene.status_code == 200
    s = scene.json()
    assert s["version"] == "original" and s["task_id"] == tid and s["up_vector"] == [0.0, -1.0, 0.0]
    assert s["splat"]["size_bytes"] == 320 and s["splat"]["sha256"] == sha((out / "export/scene.splat").read_bytes())
    assert s["points"] and s["ply"] and len(s["cameras"]) == 3
    assert s["meta"]["num_gaussians"] == 10 and s["meta"]["pose_method"] == "glomap"
    # no filesystem paths leak: not the server's, not Kaggle's
    raw = scene.text
    assert "/kaggle" not in raw and str(settings.data_dir) not in raw and settings.data_dir.as_posix() not in raw
    assert "secret_path" not in raw and "source" not in s["meta"]

    # the signed URLs work without any header (that is how splat viewers load files)
    got = client.get(s["splat"]["url"])
    assert got.status_code == 200 and got.content == (out / "export/scene.splat").read_bytes()
    assert got.headers["accept-ranges"] == "bytes"
    assert client.get(s["cameras_url"]).json()[0]["img_name"] == "000001.jpg"
    assert client.get(s["points"]["url"]).content.startswith(b"ply")

    arts = {a["kind"]: a for a in client.get(f"/projects/{pid}/artifacts", headers=auth).json()}
    assert arts["frames"]["is_dir"] and arts["frames"]["info"]["num_frames"] == 3
    assert arts["scene"]["info"]["num_images"] == 3
    assert client.get(f"/tasks/{tid}", headers=auth).json()["status"] == "done"


# ------------------------------------------------------------------ chunk protocol
def test_interrupted_upload_resumes(client, worker, running_task):
    _, tid = running_task
    data = os.urandom(10_000)                                    # 3 chunks of 4096
    u = new_upload(client, worker, tid, "splat", data[:9984])    # 312 * 32 bytes
    data = data[:9984]
    assert put(client, worker, u["upload_id"], 0, data[:4096]).status_code == 200
    # connection drops in the middle of chunk 1: the server gets only part of it
    r = put(client, worker, u["upload_id"], 1, data[4096:6000])
    assert r.status_code == 422 and "interrupted" in r.json()["detail"]

    # worker restarts: asking again for the same file returns the SAME upload with what is still missing
    again = new_upload(client, worker, tid, "splat", data)
    assert again["upload_id"] == u["upload_id"] and again["missing"] == [1, 2] and again["received"] == 1
    for i in again["missing"]:
        assert put(client, worker, u["upload_id"], i, data[i * 4096:(i + 1) * 4096]).status_code == 200
    r = finish(client, worker, u["upload_id"])
    assert r.status_code == 200 and r.json()["status"] == "complete" and r.json()["meta"]["num_gaussians"] == 312
    # and once complete, asking again says so: nothing to re-send
    assert new_upload(client, worker, tid, "splat", data)["status"] == "complete"


def test_duplicate_and_out_of_order_chunks(client, worker, running_task):
    _, tid = running_task
    data = os.urandom(32 * 400)                                  # 12800 bytes -> 4 chunks
    u = new_upload(client, worker, tid, "splat", data)
    for i in (3, 1, 0, 1, 2, 3):                                 # any order, chunk 1 and 3 twice
        r = put(client, worker, u["upload_id"], i, data[i * 4096:(i + 1) * 4096])
        assert r.status_code == 200
    assert r.json() == {"index": 3, "received": 4, "total_chunks": 4, "duplicate": True}
    assert finish(client, worker, u["upload_id"]).status_code == 200


def test_chunk_checksum_mismatch_is_rejected(client, worker, running_task):
    _, tid = running_task
    data = os.urandom(32 * 100)
    u = new_upload(client, worker, tid, "splat", data)
    r = put(client, worker, u["upload_id"], 0, data[:4096] if len(data) > 4096 else data,
            **{"X-Chunk-Sha256": "0" * 64})
    assert r.status_code == 422 and "checksum mismatch" in r.json()["detail"]
    assert client.get(f"/worker/uploads/{u['upload_id']}", headers=worker).json()["received"] == 0


def test_whole_file_checksum_failure_discards_chunks(client, worker, running_task):
    _, tid = running_task
    data = os.urandom(32 * 300)
    u = new_upload(client, worker, tid, "splat", data, claimed_sha=sha(b"something else"))
    for i in range(u["total_chunks"]):
        put(client, worker, u["upload_id"], i, data[i * 4096:(i + 1) * 4096])
    r = finish(client, worker, u["upload_id"])
    assert r.status_code == 422 and "sha256 mismatch" in r.json()["detail"]
    st = client.get(f"/worker/uploads/{u['upload_id']}", headers=worker).json()
    assert st["status"] == "open" and st["received"] == 0 and st["missing"] == list(range(u["total_chunks"]))


def test_complete_with_missing_chunks(client, worker, running_task):
    _, tid = running_task
    data = os.urandom(32 * 300)
    u = new_upload(client, worker, tid, "splat", data)
    put(client, worker, u["upload_id"], 0, data[:4096])
    r = finish(client, worker, u["upload_id"])
    assert r.status_code == 409 and r.json()["detail"]["missing"] == [1, 2]


def test_bad_chunk_requests(client, worker, running_task):
    _, tid = running_task
    data = os.urandom(32 * 200)
    u = new_upload(client, worker, tid, "splat", data)
    assert put(client, worker, u["upload_id"], 99, b"x").status_code == 422          # index out of range
    assert put(client, worker, u["upload_id"], 0, data[:4096] + b"extra").status_code == 422   # too long
    r = client.post(f"/worker/tasks/{tid}/uploads", headers=worker, json={
        "worker_id": WORKER_ID, "kind": "database", "size_bytes": 10, "sha256": "a" * 64})
    assert r.status_code == 422                                  # unknown kind -> no arbitrary destinations
    r = client.post(f"/worker/tasks/{tid}/uploads", headers=worker, json={
        "worker_id": WORKER_ID, "kind": "splat", "size_bytes": 10, "sha256": "a" * 64, "chunk_size": 200 * 2**20})
    assert r.status_code == 422                                  # chunk above the Cloudflare-safe limit


def test_upload_requires_live_lease(client, worker, running_task):
    _, tid = running_task
    data = os.urandom(320)
    u = new_upload(client, worker, tid, "splat", data)
    db = client.app.state.SessionLocal()
    db.get(Task, tid).lease_expires_at = utcnow() - timedelta(seconds=1)
    db.commit()
    db.close()
    client.get("/worker/ping", headers=worker)
    client.post("/worker/claim", json={"worker_id": "someone-else"}, headers=worker)   # re-queued + taken
    assert put(client, worker, u["upload_id"], 0, data).status_code == 409
    assert finish(client, worker, u["upload_id"]).status_code == 409


# ------------------------------------------------------------------ archive safety + validation
def _tar(entries):
    import io
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, payload, kind in entries:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(payload)
                tf.addfile(info, __import__("io").BytesIO(payload))
            elif kind == "sym":
                info.type, info.linkname = tarfile.SYMTYPE, payload
                tf.addfile(info)
            elif kind == "fifo":
                info.type = tarfile.FIFOTYPE
                tf.addfile(info)
    return buf.getvalue()


@pytest.mark.parametrize("entries, reason", [
    ([("frames/../../evil.txt", b"x", "file")], "traversal"),
    ([("/etc/passwd", b"x", "file")], "absolute"),
    ([("metrics/x.json", b"{}", "file")], "outside 'frames/'"),
    ([("frames/images/a.jpg", b"x", "file"), ("frames/link", "../../../app.db", "sym")], "escapes"),
    ([("frames/link", "/etc/passwd", "sym")], "absolute link"),
    ([("frames/pipe", None, "fifo")], "unsupported"),
])
def test_malicious_archives_rejected(client, worker, running_task, settings, entries, reason):
    _, tid = running_task
    data = _tar(entries)
    u = new_upload(client, worker, tid, "frames", data, chunk_size=len(data))
    put(client, worker, u["upload_id"], 0, data)
    r = finish(client, worker, u["upload_id"])
    assert r.status_code == 422 and reason in r.json()["detail"], r.text
    assert not (settings.data_dir / "evil.txt").exists() and not (settings.workspace_dir.parent / "evil.txt").exists()


def test_corrupt_archive_rejected(client, worker, running_task):
    _, tid = running_task
    data = b"definitely not a tar file" * 50
    assert upload(client, worker, tid, "frames", data, expect=422)["detail"].startswith("frames rejected")


@pytest.mark.parametrize("kind, mutate, reason", [
    ("splat", lambda o: (o / "export/scene.splat").write_bytes(b"\1" * 33), "multiple of 32"),
    ("cameras", lambda o: (o / "export/cameras.json").write_text("{not json"), "not valid JSON"),
    ("cameras", lambda o: (o / "export/cameras.json").write_text('[{"img_name": "a"}]'), "miss"),
    ("ply", lambda o: (o / "export/scene.ply").write_bytes(b"PK zip"), "PLY header"),
])
def test_file_validation(client, worker, running_task, out, kind, mutate, reason):
    _, tid = running_task
    mutate(out)
    rel = {"splat": "export/scene.splat", "cameras": "export/cameras.json", "ply": "export/scene.ply"}[kind]
    r = upload(client, worker, tid, kind, (out / rel).read_bytes(), expect=422)
    assert reason in r["detail"]


def test_scene_folder_validation(client, worker, running_task, out):
    _, tid = running_task
    (out / "scene/sparse/0/points3D.txt").unlink()
    r = upload(client, worker, tid, "scene", tar_bytes(out, "scene", {"scene/images": "../frames/images"}), expect=422)
    assert "points3D" in r["detail"]


def test_scene_symlink_needs_frames_first(client, worker, running_task, out):
    _, tid = running_task
    r = upload(client, worker, tid, "scene", tar_bytes(out, "scene", {"scene/images": "../frames/images"}), expect=422)
    assert "upload frames and poses before scene" in r["detail"]


def test_gzip_archive_accepted(client, worker, running_task, out):
    _, tid = running_task
    r = upload(client, worker, tid, "frames", tar_bytes(out, "frames", gz=True))
    assert r["status"] == "complete" and r["meta"]["num_frames"] == 3


# ------------------------------------------------------------------ missing artifacts / failed runs
def test_failed_run_keeps_previous_scene(client, auth, worker, settings, running_task, out):
    pid, tid = running_task
    upload_reconstruction(client, worker, tid, out)
    assert complete_task(client, worker, tid).status_code == 200
    old_splat = client.get(f"/projects/{pid}/scene", headers=auth).json()["splat"]["sha256"]

    # second run uploads a new splat, then fails: the new splat must not replace the good one
    tid2 = client.post(f"/projects/{pid}/reconstruct", headers=auth).json()["id"]
    client.post("/worker/claim", json={"worker_id": WORKER_ID}, headers=worker)
    upload(client, worker, tid2, "splat", b"\2" * 64)
    client.post(f"/worker/tasks/{tid2}/fail", headers=worker,
                json={"worker_id": WORKER_ID, "error": "boom", "retryable": False})
    assert client.get(f"/projects/{pid}/scene", headers=auth).json()["splat"]["sha256"] == old_splat


def test_missing_files_and_paths(client, auth, worker, settings, running_task, out):
    pid, tid = running_task
    assert client.get(f"/projects/{pid}/scene", headers=auth).status_code == 404
    upload_reconstruction(client, worker, tid, out)
    complete_task(client, worker, tid)
    token, _ = create_file_token(pid, settings.jwt_secret, 5)
    for bad in ("app.db", "..%2F..%2Fapp.db", "metrics/progress.jsonl", ".worker_token", "export/nothing.splat",
                "export%2F..%2F..%2F..%2Fapp.db", "frames/images/..%2F..%2F..%2F.jwt_secret"):
        assert client.get(f"/projects/{pid}/files/{bad}?t={token}").status_code == 404, bad
    # raw ../ segments are normalised by the client before sending; whatever route they land on, nothing leaks
    for bad in ("../../app.db", "frames/../../../app.db", "../../.jwt_secret"):
        r = client.get(f"/projects/{pid}/files/{bad}?t={token}")
        assert r.status_code != 200 and b"SQLite" not in r.content, bad
    # artifact registered but its file deleted from disk
    (settings.workspace_dir / pid / "export" / "scene.splat").unlink()
    assert client.get(f"/projects/{pid}/files/export/scene.splat?t={token}").status_code == 404
    assert client.get(f"/projects/{pid}/scene", headers=auth).status_code == 404


def test_file_access_control(client, auth, worker, settings, running_task, out):
    pid, tid = running_task
    upload_reconstruction(client, worker, tid, out)
    complete_task(client, worker, tid)
    f = f"/projects/{pid}/files/export/scene.splat"
    assert client.get(f).status_code == 401                                       # no auth at all
    assert client.get(f, headers=auth).status_code == 200                         # owner's JWT
    assert client.get(f, headers=signup(client, "other@example.com")).status_code == 404
    other_tok, _ = create_file_token("another-project", settings.jwt_secret, 5)
    assert client.get(f"{f}?t={other_tok}").status_code == 404                    # token for another project
    expired, _ = create_file_token(pid, settings.jwt_secret, -1)
    assert client.get(f"{f}?t={expired}").status_code == 404
    forged, _ = create_file_token(pid, "x" * 40, 5)
    assert client.get(f"{f}?t={forged}").status_code == 404
    assert client.get(f"/worker/projects/{pid}/files/export/scene.splat", headers=worker).status_code == 200
    assert client.get(f"/worker/projects/{pid}/files/export/scene.splat", headers=auth).status_code == 401


# ------------------------------------------------------------------ large file + HTTP Range
def test_large_file_chunked_upload_and_range_download(client, auth, worker, running_task, out, settings):
    pid, tid = running_task
    big = b"ply\n" + os.urandom(64 * 2**20)                       # ~64 MB scene.ply
    upload_reconstruction(client, worker, tid, out)               # includes a small scene.ply ...
    r = upload(client, worker, tid, "ply", big, chunk_size=8 * 2**20)   # ... replaced by the big one
    assert r["total_chunks"] == 9 and r["status"] == "complete"
    assert complete_task(client, worker, tid).status_code == 200
    assert (settings.workspace_dir / pid / "export" / "scene.ply").stat().st_size == len(big)

    url = client.get(f"/projects/{pid}/scene", headers=auth).json()["ply"]["url"]
    n = len(big)
    full = client.get(url)
    assert full.status_code == 200 and hashlib.sha256(full.content).hexdigest() == sha(big)

    r = client.get(url, headers={"Range": "bytes=0-1023"})
    assert r.status_code == 206 and r.content == big[:1024] and r.headers["content-range"] == f"bytes 0-1023/{n}"
    mid = n // 2
    r = client.get(url, headers={"Range": f"bytes={mid}-{mid + 99}"})
    assert r.status_code == 206 and r.content == big[mid:mid + 100]
    r = client.get(url, headers={"Range": "bytes=-500"})                           # last 500 bytes
    assert r.status_code == 206 and r.content == big[-500:]
    r = client.get(url, headers={"Range": f"bytes={n - 10}-"})                     # open-ended (resume)
    assert r.status_code == 206 and r.content == big[-10:]
    assert client.get(url, headers={"Range": f"bytes={n + 10}-{n + 20}"}).status_code == 416

    # resume a download that stopped at an arbitrary byte, then check the whole thing
    cut = 12_345_678
    rest = client.get(url, headers={"Range": f"bytes={cut}-"})
    assert rest.status_code == 206 and hashlib.sha256(big[:cut] + rest.content).hexdigest() == sha(big)


def test_reupload_after_different_content_is_not_skipped(client, auth, worker, settings, running_task, out):
    """A, then B, then A again: the third upload must really be transferred and A must end up stored."""
    pid, tid = running_task
    a, b = b"" * 320, b"" * 320
    upload(client, worker, tid, "splat", a)
    upload(client, worker, tid, "splat", b)
    again = new_upload(client, worker, tid, "splat", a)
    assert again["status"] == "open" and again["missing"] == [0]
    upload(client, worker, tid, "splat", a)
    upload_reconstruction(client, worker, tid, out)
    upload(client, worker, tid, "splat", a)                     # same as staged now -> skipped, still a
    assert complete_task(client, worker, tid).status_code == 200
    assert (settings.workspace_dir / pid / "export" / "scene.splat").read_bytes() == a
