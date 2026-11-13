"""
Plays the Kaggle worker so the frontend can be tested against the REAL backend without a GPU.

It signs up a user, creates a project, uploads a generated video, submits a reconstruction,
then claims that task as the worker and uploads a synthetic room (a valid .splat, a coloured
points.ply, training cameras that really look at it, frames) exactly the way the backend's own tests do.

    python tests/live/seed_scene.py --api http://localhost:8000 --worker-token-file ../backend/data/.worker_token

Prints one JSON line: {"email", "password", "token", "project_id", "task_id"}.
"""
import argparse, hashlib, io, json, math, os, struct, subprocess, sys, tarfile, tempfile, uuid
from pathlib import Path

import httpx
import numpy as np

WORKER = "seed-worker"


def sha(b): return hashlib.sha256(b).hexdigest()


def make_room(n=60000, seed=1):
    """Gaussians on the walls/floor/ceiling of a 5 x 2.7 x 4 m room (COLMAP world: +y DOWN) plus a 'sofa' box."""
    rng = np.random.default_rng(seed)
    X, Y0, Y1, Z = 2.5, -1.5, 1.2, 2.0  # x half-width, ceiling y, floor y, z half-depth
    pts, cols = [], []

    def face(count, fixed_axis, fixed_val, ranges, color):
        p = np.zeros((count, 3))
        for ax in range(3):
            p[:, ax] = fixed_val if ax == fixed_axis else rng.uniform(*ranges[ax], count)
        c = np.clip(np.array(color) + rng.normal(0, 14, (count, 3)), 0, 255)
        pts.append(p); cols.append(c)

    R = [(-X, X), (Y0, Y1), (-Z, Z)]
    face(n // 5, 1, Y1, R, (80, 160, 110))    # floor: green
    face(n // 6, 1, Y0, R, (245, 240, 225))   # ceiling
    face(n // 6, 0, -X, R, (60, 110, 190))    # left wall: blue
    face(n // 6, 0, X, R, (245, 190, 70))     # right wall: yellow
    face(n // 6, 2, Z, R, (225, 225, 215))
    face(n // 6, 2, -Z, R, (225, 225, 215))
    # sofa: box against the +z wall, dark red so edits have something obvious to target
    S = [(-1.2, 1.2), (0.3, 1.2), (1.2, 1.9)]
    for ax, val in ((1, 0.3), (2, 1.2)):
        face(n // 20, ax, val, S, (170, 60, 55))
    face(n // 20, 0, -1.2, S, (170, 60, 55)); face(n // 20, 0, 1.2, S, (170, 60, 55))
    return np.vstack(pts).astype(np.float32), np.vstack(cols).astype(np.uint8)


def splat_bytes(pos, col, rng=np.random.default_rng(2)):
    n = len(pos)
    out = bytearray()
    scale = rng.uniform(0.012, 0.03, (n, 3)).astype(np.float32)
    rgba = np.concatenate([col, np.full((n, 1), 255, np.uint8)], axis=1)
    rot = np.tile(np.array([255, 128, 128, 128], np.uint8), (n, 1))  # identity quaternion (w,x,y,z) packed
    for i in range(n):
        out += pos[i].tobytes() + scale[i].tobytes() + rgba[i].tobytes() + rot[i].tobytes()
    return bytes(out)


def ply_bytes(pos, col):
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(pos)).encode()
    body = b"".join(struct.pack("<fffBBB", *pos[i], *col[i]) for i in range(len(pos)))
    return head + body


def cameras(n=24, w=960, h=540, fov_deg=70):
    fy = (h / 2) / math.tan(math.radians(fov_deg) / 2)
    out = []
    for i in range(n):
        th = 2 * math.pi * i / n
        f = np.array([math.sin(th), 0.0, math.cos(th)])          # forward
        d = np.array([0.0, 1.0, 0.0])                            # image-down = world +y
        r = np.cross(d, f)                                       # image-right
        Rm = np.stack([r, d, f], axis=1)                         # camera-to-world, columns = axes
        out.append({"id": i + 1, "img_name": f"{i + 1:06d}.jpg", "width": w, "height": h,
                    "position": [0.0, 0.2, 0.0], "rotation": Rm.round(6).tolist(),
                    "fx": fy, "fy": fy, "cx": w / 2, "cy": h / 2})
    return out


def tar_bytes(root: Path, top, symlinks=None):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        tf.add(root / top, arcname=top)
        for name, target in (symlinks or {}).items():
            info = tarfile.TarInfo(name); info.type, info.linkname = tarfile.SYMTYPE, target; tf.addfile(info)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--worker-token-file", required=True)
    ap.add_argument("--email", default=f"seed-{uuid.uuid4().hex[:8]}@example.com")
    ap.add_argument("--password", default="password123")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--drain", action="store_true", help="fail other users' pending tasks that are ahead in the queue (test backends only!)")
    a = ap.parse_args()
    wtoken = Path(a.worker_token_file).read_text().strip()
    wh = {"Authorization": f"Bearer {wtoken}"}
    c = httpx.Client(base_url=a.api, timeout=120)

    r = c.post("/auth/signup", json={"email": a.email, "password": a.password}); r.raise_for_status()
    token = r.json()["access_token"]; uh = {"Authorization": f"Bearer {token}"}
    pid = c.post("/projects", json={"name": "Seeded demo room"}, headers=uh).json()["id"]

    tmp = Path(tempfile.mkdtemp())
    video = tmp / "room.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=8:size=640x360:rate=10", "-pix_fmt", "yuv420p", str(video)], check=True)
    with open(video, "rb") as fh:
        c.post(f"/projects/{pid}/video", files={"file": ("room.mp4", fh, "video/mp4")}, headers=uh).raise_for_status()
    task = c.post(f"/projects/{pid}/reconstruct", json={}, headers=uh).json()
    for _ in range(50):
        claim = c.post("/worker/claim", json={"worker_id": WORKER}, headers=wh); claim.raise_for_status()
        tid = claim.json()["task"]["id"]
        if tid == task["id"]: break
        if not a.drain:
            sys.exit(f"The queue has another pending task ({tid}) ahead of ours. Re-run with --drain on a LOCAL/TEST backend to clear it.")
        c.post(f"/worker/tasks/{tid}/fail", headers=wh, json={"worker_id": WORKER, "error": "drained by seed_scene.py", "retryable": False})
    else:
        sys.exit("could not claim our task")

    root = tmp / "proj"
    (root / "frames" / "images").mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=960x540:rate=1", "-frames:v", str(a.frames),
                    "-start_number", "1", str(root / "frames" / "images" / "%06d.jpg")], check=True)
    (root / "frames" / "frames_index.json").write_text(json.dumps({"kept": a.frames, "frames": []}))
    sp = root / "scene" / "sparse" / "0"; sp.mkdir(parents=True)
    for s in ("cameras", "images", "points3D"): (sp / f"{s}.txt").write_text(f"# {s}\n")
    (root / "scene" / "chosen.json").write_text(json.dumps({"method": "synthetic"}))
    pos, col = make_room()
    (root / "scene" / "points.ply").write_bytes(ply_bytes(pos[::12], col[::12]))
    (root / "export").mkdir()
    (root / "export" / "scene.splat").write_bytes(splat_bytes(pos, col))
    (root / "export" / "scene.ply").write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    cams = cameras(a.frames)
    (root / "export" / "cameras.json").write_text(json.dumps(cams))
    (root / "export" / "scene_meta.json").write_text(json.dumps({
        "num_gaussians": len(pos), "splat_mb": round(len(pos) * 32 / 1e6, 2), "trainer": "synthetic", "iters": 0, "cameras": "cameras.json",
        "num_cameras": a.frames, "bbox_min": [-2.5, -1.5, -2.0], "bbox_max": [2.5, 1.2, 2.0], "median_position": [0.0, 0.0, 0.0],
        "up_vector": [0.0, -1.0, 0.0]}))

    def upload(kind, data, chunk=4 * 1024 * 1024):
        u = c.post(f"/worker/tasks/{tid}/uploads", headers=wh, json={"worker_id": WORKER, "kind": kind, "size_bytes": len(data), "sha256": sha(data), "chunk_size": chunk})
        u.raise_for_status(); u = u.json()
        for i in u["missing"]:
            part = data[i * u["chunk_size"]:(i + 1) * u["chunk_size"]]
            c.put(f"/worker/uploads/{u['upload_id']}/chunks/{i}", content=part, headers={**wh, "X-Worker-Id": WORKER, "X-Chunk-Sha256": sha(part)}).raise_for_status()
        c.post(f"/worker/uploads/{u['upload_id']}/complete", headers=wh, json={"worker_id": WORKER}).raise_for_status()

    c.post(f"/tasks/{tid}/progress", headers=wh, json={"status": "running", "stage": "training", "progress": 60}).raise_for_status()
    upload("frames", tar_bytes(root, "frames"))
    upload("scene", tar_bytes(root, "scene", {"scene/images": "../frames/images"}))
    for kind, rel in (("splat", "export/scene.splat"), ("ply", "export/scene.ply"), ("cameras", "export/cameras.json"), ("scene_meta", "export/scene_meta.json")):
        upload(kind, (root / rel).read_bytes())
    c.post(f"/worker/tasks/{tid}/complete", headers=wh, json={"worker_id": WORKER, "result": {}}).raise_for_status()
    print(json.dumps({"email": a.email, "password": a.password, "token": token, "project_id": pid, "task_id": tid}))


if __name__ == "__main__":
    sys.exit(main())
