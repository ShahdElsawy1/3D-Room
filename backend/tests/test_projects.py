from conftest import make_video, needs_ffmpeg, signup

from app.models import Task


def test_project_crud_and_isolation(client, auth):
    r = client.post("/projects", json={"name": "Living room"}, headers=auth)
    assert r.status_code == 201
    p = r.json()
    assert p["name"] == "Living room" and p["has_video"] is False and p["video"] is None
    assert p["created_at"].endswith("Z")

    client.post("/projects", json={"name": "Kitchen"}, headers=auth)
    names = [x["name"] for x in client.get("/projects", headers=auth).json()]
    assert names == ["Kitchen", "Living room"]                       # newest first

    r = client.patch(f"/projects/{p['id']}", json={"name": "Lounge", "description": "test"}, headers=auth)
    assert r.json()["name"] == "Lounge" and r.json()["description"] == "test"

    # another user sees nothing and gets 404 (not 403) on direct access
    other = signup(client, "b@example.com")
    assert client.get("/projects", headers=other).json() == []
    assert client.get(f"/projects/{p['id']}", headers=other).status_code == 404
    assert client.patch(f"/projects/{p['id']}", json={"name": "x"}, headers=other).status_code == 404
    assert client.delete(f"/projects/{p['id']}", headers=other).status_code == 404

    assert client.delete(f"/projects/{p['id']}", headers=auth).status_code == 204
    assert client.get(f"/projects/{p['id']}", headers=auth).status_code == 404


def test_project_validation(client, auth):
    assert client.post("/projects", json={"name": ""}, headers=auth).status_code == 422
    assert client.get("/projects/doesnotexist", headers=auth).status_code == 404


@needs_ffmpeg
def test_video_upload_store_and_stream(client, auth, settings, video_dir):
    pid = client.post("/projects", json={"name": "Room"}, headers=auth).json()["id"]
    vid = make_video(video_dir / "room.mp4", 4)
    with vid.open("rb") as f:
        r = client.post(f"/projects/{pid}/video", files={"file": ("my room.mp4", f, "video/mp4")}, headers=auth)
    assert r.status_code == 201, r.text
    v = r.json()["video"]
    assert (v["width"], v["height"]) == (640, 360)
    assert abs(v["duration_s"] - 4) < 0.5 and v["codec"] == "h264" and v["size_bytes"] == vid.stat().st_size
    assert any("recommended" in w for w in v["warnings"])            # 4 s is outside 2-5 min -> warning, not error

    stored = settings.workspace_dir / pid / "video" / "input.mp4"     # Person A's layout: <project>/video/input.<ext>
    assert stored.exists() and stored.stat().st_size == vid.stat().st_size

    proj = client.get(f"/projects/{pid}", headers=auth).json()
    assert proj["has_video"] and proj["video"]["filename"] == "my room.mp4"

    full = client.get(f"/projects/{pid}/video", headers=auth)
    assert full.status_code == 200 and full.content == vid.read_bytes()
    part = client.get(f"/projects/{pid}/video", headers={**auth, "Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100     # browser <video> seeking works

    # re-upload replaces the file (a .mov this time) instead of leaving two inputs around
    mov = make_video(video_dir / "room.mov", 3)
    with mov.open("rb") as f:
        assert client.post(f"/projects/{pid}/video", files={"file": ("r.mov", f)}, headers=auth).status_code == 201
    assert sorted(x.name for x in stored.parent.iterdir()) == ["input.mov"]


@needs_ffmpeg
def test_video_rejections(client, auth, settings, video_dir):
    pid = client.post("/projects", json={"name": "Room"}, headers=auth).json()["id"]

    r = client.post(f"/projects/{pid}/video", files={"file": ("notes.txt", b"hello")}, headers=auth)
    assert r.status_code == 415

    r = client.post(f"/projects/{pid}/video", files={"file": ("fake.mp4", b"this is not a video" * 100)}, headers=auth)
    assert r.status_code == 422 and "Invalid video" in r.json()["detail"]

    short = make_video(video_dir / "short.mp4", 1)                   # below video_min_seconds=2
    with short.open("rb") as f:
        r = client.post(f"/projects/{pid}/video", files={"file": ("short.mp4", f)}, headers=auth)
    assert r.status_code == 422 and "minimum" in r.json()["detail"]

    r = client.post(f"/projects/{pid}/video", files={"file": ("big.mp4", b"\0" * (6 * 1024 * 1024))}, headers=auth)
    assert r.status_code == 413                                      # max_upload_mb=5 in the test settings

    assert not (settings.workspace_dir / pid / "video").exists() or not any((settings.workspace_dir / pid / "video").iterdir())
    assert list(settings.tmp_dir.iterdir()) == []                    # temp files cleaned up on every failure path
    assert client.get(f"/projects/{pid}/video", headers=auth).status_code == 404


@needs_ffmpeg
def test_video_locked_while_task_active(client, auth, video_dir):
    pid = client.post("/projects", json={"name": "Room"}, headers=auth).json()["id"]
    db = client.app.state.SessionLocal()
    db.add(Task(project_id=pid, type="reconstruct", status="running"))
    db.commit()
    db.close()
    vid = make_video(video_dir / "locked.mp4", 3)
    with vid.open("rb") as f:
        assert client.post(f"/projects/{pid}/video", files={"file": ("v.mp4", f)}, headers=auth).status_code == 409
    assert client.delete(f"/projects/{pid}", headers=auth).status_code == 409
    assert client.get(f"/projects/{pid}", headers=auth).json()["latest_task_status"] == "running"


def test_delete_removes_files(client, auth, settings):
    pid = client.post("/projects", json={"name": "Room"}, headers=auth).json()["id"]
    d = settings.workspace_dir / pid
    (d / "export").mkdir(parents=True)
    (d / "export" / "scene.splat").write_bytes(b"x")
    assert client.delete(f"/projects/{pid}", headers=auth).status_code == 204
    assert not d.exists()
