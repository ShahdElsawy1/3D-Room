from datetime import timedelta

import pytest
from conftest import signup

from app.models import Artifact, Task, utcnow


@pytest.fixture
def worker(settings):
    return {"Authorization": f"Bearer {settings.worker_token}"}


@pytest.fixture
def project_with_video(client, auth, settings):
    """A project whose video is registered without running ffprobe (upload itself is covered in test_projects)."""
    pid = client.post("/projects", json={"name": "Room"}, headers=auth).json()["id"]
    vdir = settings.workspace_dir / pid / "video"
    vdir.mkdir(parents=True)
    (vdir / "input.mp4").write_bytes(b"fake video bytes")
    db = client.app.state.SessionLocal()
    from app.models import Project
    p = db.get(Project, pid)
    p.video_path = "video/input.mp4"
    db.add(Artifact(project_id=pid, kind="video", path="video/input.mp4", size_bytes=16, sha256="abc"))
    db.commit()
    db.close()
    return pid


def set_task(client, task_id, **fields):
    db = client.app.state.SessionLocal()
    t = db.get(Task, task_id)
    for k, v in fields.items():
        setattr(t, k, v)
    db.commit()
    db.close()


def test_reconstruct_requires_video_and_single_active_task(client, auth, project_with_video):
    empty = client.post("/projects", json={"name": "Empty"}, headers=auth).json()["id"]
    assert client.post(f"/projects/{empty}/reconstruct", headers=auth).status_code == 409

    r = client.post(f"/projects/{project_with_video}/reconstruct", json={"method": "vggt", "train_mode": "quick"},
                    headers=auth)
    assert r.status_code == 202
    t = r.json()
    assert t["status"] == "pending" and t["stage"] == "queued" and t["progress"] == 0 and t["queue_position"] == 1
    assert t["params"] == {"method": "vggt", "train_mode": "quick"}
    assert [s["name"] for s in t["stages"]] == ["extracting frames", "estimating poses", "training", "exporting"]

    assert client.post(f"/projects/{project_with_video}/reconstruct", headers=auth).status_code == 409
    assert client.get(f"/projects/{project_with_video}", headers=auth).json()["latest_task_status"] == "pending"


def test_reconstruct_param_validation(client, auth, project_with_video):
    for bad in ({"method": "nerf"}, {"long_side": 1920}, {"long_side": 320}, {"fps": "10"}):
        assert client.post(f"/projects/{project_with_video}/reconstruct", json=bad, headers=auth).status_code == 422


def test_worker_routes_need_worker_token(client, auth, project_with_video):
    tid = client.post(f"/projects/{project_with_video}/reconstruct", headers=auth).json()["id"]
    body = {"status": "running", "stage": "training", "progress": 50}
    assert client.post("/worker/claim", json={"worker_id": "k"}).status_code == 401
    assert client.post("/worker/claim", json={"worker_id": "k"}, headers=auth).status_code == 401   # user JWT is not enough
    assert client.post(f"/tasks/{tid}/progress", json=body).status_code == 401
    assert client.post(f"/tasks/{tid}/progress", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(f"/worker/tasks/{tid}/video").status_code == 401


def test_full_happy_path(client, auth, worker, project_with_video):
    tid = client.post(f"/projects/{project_with_video}/reconstruct", json={"method": "glomap", "long_side": 960},
                      headers=auth).json()["id"]

    r = client.post("/worker/claim", json={"worker_id": "kaggle-1"}, headers=worker)
    assert r.status_code == 200
    c = r.json()
    assert c["task"]["id"] == tid and c["task"]["status"] == "running" and c["task"]["attempts"] == 1
    assert c["pipeline_args"] == ["--method", "glomap", "--long-side", "960"]
    assert c["video_sha256"] == "abc" and c["video_filename"] == "input.mp4"
    assert client.get(c["video_url"], headers=worker).content == b"fake video bytes"
    assert client.post("/worker/claim", json={"worker_id": "kaggle-2"}, headers=worker).status_code == 204

    # exactly the records ML_Work/common/progress.py posts (incl. its extra "t" field)
    for stage, pct in (("extracting frames", 7.5), ("estimating poses", 30.0), ("training", 67.5), ("exporting", 95.0)):
        r = client.post(f"/tasks/{tid}/progress", headers=worker,
                        json={"t": 1.7e9, "status": "running", "stage": stage, "progress": pct, "error": None})
        assert r.json() == {"ok": True, "cancel": False}
        t = client.get(f"/tasks/{tid}", headers=auth).json()
        assert (t["stage"], t["progress"], t["status"]) == (stage, pct, "running")

    # Progress.done() -> pipeline finished, but results not stored yet: stays running, capped at 99
    client.post(f"/tasks/{tid}/progress", headers=worker,
                json={"status": "done", "stage": "exporting", "progress": 100, "error": None})
    t = client.get(f"/tasks/{tid}", headers=auth).json()
    assert t["status"] == "running" and t["progress"] == 99.0

    hb = client.post(f"/worker/tasks/{tid}/heartbeat", json={"worker_id": "kaggle-1", "stage": "uploading results"},
                     headers=worker).json()
    assert hb["status"] == "running" and hb["cancel"] is False and hb["lease_expires_at"].endswith("Z")

    r = client.post(f"/worker/tasks/{tid}/complete", json={"worker_id": "kaggle-1", "result": {"num_gaussians": 1000}},
                    headers=worker)
    assert r.status_code == 200
    t = client.get(f"/tasks/{tid}", headers=auth).json()
    assert (t["status"], t["progress"], t["result"]) == ("done", 100.0, {"num_gaussians": 1000})
    assert t["finished_at"] and t["error"] is None
    # a finished task accepts no more progress
    assert client.post(f"/tasks/{tid}/progress", headers=worker,
                       json={"status": "running", "stage": "training", "progress": 50}).status_code == 409


def test_other_worker_cannot_touch_leased_task(client, auth, worker, project_with_video):
    tid = client.post(f"/projects/{project_with_video}/reconstruct", headers=auth).json()["id"]
    client.post("/worker/claim", json={"worker_id": "a"}, headers=worker)
    assert client.post(f"/worker/tasks/{tid}/complete", json={"worker_id": "b"}, headers=worker).status_code == 409
    assert client.post(f"/worker/tasks/{tid}/heartbeat", json={"worker_id": "b"}, headers=worker).status_code == 409


def test_failure_retries_then_fails(client, auth, worker, project_with_video):
    tid = client.post(f"/projects/{project_with_video}/reconstruct", headers=auth).json()["id"]   # max_attempts = 2
    client.post("/worker/claim", json={"worker_id": "w"}, headers=worker)
    # the pipeline's own failure report is recorded but the worker decides about retrying
    client.post(f"/tasks/{tid}/progress", headers=worker,
                json={"status": "failed", "stage": "estimating poses", "progress": 15, "error": "CUDA OOM"})
    t = client.get(f"/tasks/{tid}", headers=auth).json()
    assert t["status"] == "running" and t["error"] == "CUDA OOM" and t["stage"] == "estimating poses"

    r = client.post(f"/worker/tasks/{tid}/fail", json={"worker_id": "w", "error": "CUDA OOM"}, headers=worker).json()
    assert r == {"ok": True, "status": "pending", "attempts": 1, "max_attempts": 2}
    t = client.get(f"/tasks/{tid}", headers=auth).json()
    assert t["status"] == "pending" and t["error"] == "CUDA OOM"          # user still sees why it is retrying

    assert client.post("/worker/claim", json={"worker_id": "w"}, headers=worker).json()["task"]["attempts"] == 2
    r = client.post(f"/worker/tasks/{tid}/fail", json={"worker_id": "w", "error": "CUDA OOM again"}, headers=worker)
    assert r.json()["status"] == "failed"
    t = client.get(f"/tasks/{tid}", headers=auth).json()
    assert t["status"] == "failed" and t["error"] == "CUDA OOM again" and t["finished_at"]
    # project is free again: the user can resubmit
    assert client.post(f"/projects/{project_with_video}/reconstruct", headers=auth).status_code == 202


def test_non_retryable_failure_is_final(client, auth, worker, project_with_video):
    tid = client.post(f"/projects/{project_with_video}/reconstruct", headers=auth).json()["id"]
    client.post("/worker/claim", json={"worker_id": "w"}, headers=worker)
    r = client.post(f"/worker/tasks/{tid}/fail", json={"worker_id": "w", "error": "bad video", "retryable": False},
                    headers=worker)
    assert r.json()["status"] == "failed"


def test_lease_expiry_requeues_then_fails(client, auth, worker, project_with_video):
    tid = client.post(f"/projects/{project_with_video}/reconstruct", headers=auth).json()["id"]
    client.post("/worker/claim", json={"worker_id": "dead-session"}, headers=worker)
    set_task(client, tid, lease_expires_at=utcnow() - timedelta(seconds=1))     # Kaggle session died silently

    t = client.get(f"/tasks/{tid}", headers=auth).json()
    assert t["status"] == "pending" and "stopped responding" in t["error"]
    # the dead worker can no longer report on it
    assert client.post(f"/worker/tasks/{tid}/complete", json={"worker_id": "dead-session"},
                       headers=worker).status_code == 409

    assert client.post("/worker/claim", json={"worker_id": "new"}, headers=worker).json()["task"]["attempts"] == 2
    set_task(client, tid, lease_expires_at=utcnow() - timedelta(seconds=1))
    t = client.get(f"/tasks/{tid}", headers=auth).json()
    assert t["status"] == "failed" and t["finished_at"]


def test_progress_and_heartbeat_renew_lease(client, auth, worker, project_with_video, settings):
    tid = client.post(f"/projects/{project_with_video}/reconstruct", headers=auth).json()["id"]
    client.post("/worker/claim", json={"worker_id": "w"}, headers=worker)
    set_task(client, tid, lease_expires_at=utcnow() + timedelta(seconds=5))
    client.post(f"/tasks/{tid}/progress", headers=worker, json={"stage": "training", "progress": 50})
    db = client.app.state.SessionLocal()
    assert db.get(Task, tid).lease_expires_at > utcnow() + timedelta(seconds=settings.lease_seconds - 30)
    db.close()


def test_cancel_pending_and_running(client, auth, worker, project_with_video):
    tid = client.post(f"/projects/{project_with_video}/reconstruct", headers=auth).json()["id"]
    assert client.post(f"/tasks/{tid}/cancel", headers=auth).json()["status"] == "cancelled"
    assert client.post("/worker/claim", json={"worker_id": "w"}, headers=worker).status_code == 204
    assert client.post(f"/tasks/{tid}/cancel", headers=auth).status_code == 409

    tid = client.post(f"/projects/{project_with_video}/reconstruct", headers=auth).json()["id"]
    client.post("/worker/claim", json={"worker_id": "w"}, headers=worker)
    client.post(f"/tasks/{tid}/cancel", headers=auth)
    # the running worker is told to stop, through both channels
    assert client.post(f"/worker/tasks/{tid}/heartbeat", json={"worker_id": "w"}, headers=worker).json()["cancel"]
    assert client.post(f"/tasks/{tid}/progress", headers=worker,
                       json={"stage": "training", "progress": 60}).json()["cancel"] is True
    assert client.post(f"/worker/tasks/{tid}/complete", json={"worker_id": "w"}, headers=worker).status_code == 409
    assert client.get(f"/tasks/{tid}", headers=auth).json()["status"] == "cancelled"


def test_task_isolation_and_history(client, auth, project_with_video):
    tid = client.post(f"/projects/{project_with_video}/reconstruct", headers=auth).json()["id"]
    other = signup(client, "other@example.com")
    assert client.get(f"/tasks/{tid}", headers=other).status_code == 404
    assert client.post(f"/tasks/{tid}/cancel", headers=other).status_code == 404
    assert client.post(f"/projects/{project_with_video}/reconstruct", headers=other).status_code == 404
    hist = client.get(f"/projects/{project_with_video}/tasks", headers=auth).json()
    assert [t["id"] for t in hist] == [tid]


def test_queue_order_and_type_filter(client, auth, worker, settings):
    ids = []
    for name in ("first", "second"):
        pid = client.post("/projects", json={"name": name}, headers=auth).json()["id"]
        db = client.app.state.SessionLocal()
        from app.models import Project
        db.get(Project, pid).video_path = "video/input.mp4"
        db.commit()
        db.close()
        ids.append(client.post(f"/projects/{pid}/reconstruct", headers=auth).json()["id"])
    assert client.get(f"/tasks/{ids[1]}", headers=auth).json()["queue_position"] == 2
    assert client.post("/worker/claim", json={"worker_id": "w", "types": ["rebuild"]}, headers=worker).status_code == 204
    assert client.post("/worker/claim", json={"worker_id": "w"}, headers=worker).json()["task"]["id"] == ids[0]
    assert client.get(f"/tasks/{ids[1]}", headers=auth).json()["queue_position"] == 1


def test_worker_ping(client, worker):
    assert client.get("/worker/ping", headers=worker).json()["ok"] is True
    assert client.get("/worker/ping", headers={"Authorization": "Bearer nope"}).status_code == 401
