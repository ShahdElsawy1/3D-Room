#!/usr/bin/env python3
"""Pull-based Kaggle worker (spec 4.3): polls the backend, runs Person A's pipeline, uploads the results.

    python kaggle_worker.py --backend-url https://<tunnel>.trycloudflare.com --ml-dir /kaggle/working/repo/ML_Work
    (token from env W1_WORKER_TOKEN, or --token)

One task:
    claim -> download video (Range-resumable, sha256 checked) -> run ML_Work/run_pipeline.py as a child process
    (it POSTs real stage progress itself) -> heartbeat thread renews the lease and stops the pipeline if the user
    cancels or the lease is lost -> upload artifacts (resumable chunks, Person A's layout) -> complete | fail

Nothing in ML_Work is modified or imported; the pipeline is only run. Needs: Python 3.10+, requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
from collections import deque
from pathlib import Path

import requests

# ------------------------------------------------------------------ what gets uploaded, in this order
# (kind, path in A's layout, is_dir, required). Small viewer files go first so a slow tunnel delivers the scene
# early in the upload phase; the big optional ones go last.
UPLOADS = [
    ("frames", "frames", True, True),
    ("splat", "export/scene.splat", False, True),
    ("cameras", "export/cameras.json", False, True),
    ("scene_meta", "export/scene_meta.json", False, True),
    ("scene", "scene", True, True),
    ("poses", "poses", True, False),
    ("metrics", "metrics", True, False),
    ("manifest", "manifest.json", False, False),
    ("ply", "export/scene.ply", False, False),        # ~250 MB: only with --upload-ply
    ("gs", "gs", True, False),                         # checkpoint(s): only with --upload-checkpoint
]
NUMERIC = (int, float, bool)


def log(msg: str) -> None:
    print(f"[worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class LeaseLost(Exception):
    """The backend no longer considers this worker the owner of the task (cancelled / re-queued)."""


class Api:
    """requests wrapper: retries connection errors and 5xx (tunnel hiccups) with backoff, never 4xx."""

    def __init__(self, base: str, token: str, retries: int = 6):
        self.base = base.rstrip("/")
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"
        self.retries = retries

    def call(self, method: str, path: str, ok=(200,), **kw) -> requests.Response:
        kw.setdefault("timeout", (15, 300))
        delay = 2.0
        for attempt in range(1, self.retries + 1):
            try:
                r = self.s.request(method, self.base + path, **kw)
                if r.status_code < 500:
                    if r.status_code == 409 and "not leased" in r.text:
                        raise LeaseLost(r.text[:300])
                    if r.status_code not in ok:
                        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
                    return r
                err = f"HTTP {r.status_code}"
            except (requests.ConnectionError, requests.Timeout) as e:
                err = e.__class__.__name__
            if attempt == self.retries:
                raise RuntimeError(f"{method} {path} failed after {attempt} tries ({err})")
            log(f"{method} {path}: {err}, retry {attempt}/{self.retries - 1} in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise AssertionError("unreachable")


# ------------------------------------------------------------------ helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def download(api: Api, url: str, dest: Path, size: int | None, sha: str | None) -> None:
    """Resumable download (HTTP Range) with sha256 check."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        have = dest.stat().st_size if dest.exists() else 0
        if size is None or have < size:
            headers = {"Range": f"bytes={have}-"} if have else {}
            r = api.call("GET", url, ok=(200, 206), headers=headers, stream=True)
            mode = "ab" if r.status_code == 206 else "wb"
            with dest.open(mode) as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
        if sha is None or sha256_file(dest) == sha:
            return
        log("video checksum mismatch, downloading again")
        dest.unlink()
    raise RuntimeError("video download failed checksum 3 times")


def make_tar(project: Path, rel: str, out: Path) -> Path:
    """tar of project/rel with entries named rel/... Symlinks are followed (A's scene/images is a symlink on
    Kaggle), so the archive holds real files. poses/<method>/images is skipped: it is a link or copy of the
    frames, and the images actually used for training are already in scene/images."""
    def keep(ti: tarfile.TarInfo):
        parts = ti.name.split("/")
        if rel == "poses" and len(parts) >= 3 and parts[2] == "images":
            return None
        return ti

    with tarfile.open(out, "w", dereference=True) as tf:
        tf.add(project / rel, arcname=rel, filter=keep)
    return out


def kill_tree(proc: subprocess.Popen) -> None:
    """Stop the pipeline AND its stage subprocesses (they hold the GPU)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


def stage_metrics(metrics_dir: Path) -> dict:
    """Runtime/VRAM per stage from A's metrics/*.json, numbers only (A's files also hold /kaggle paths)."""
    out = {}
    for f in sorted(metrics_dir.glob("s*_*.json")):
        try:
            d = json.loads(f.read_text())
        except ValueError:
            continue
        out[f.stem] = {k: v for k, v in d.items() if isinstance(v, NUMERIC) and k != "gpu_baseline_mb"}
    return out


# ------------------------------------------------------------------ heartbeat
class Heartbeat(threading.Thread):
    def __init__(self, api: Api, task_id: str, worker_id: str, every: float):
        super().__init__(daemon=True)
        self.api, self.task_id, self.worker_id, self.every = api, task_id, worker_id, every
        self.stop_evt = threading.Event()
        self.cancelled = threading.Event()     # user cancelled or lease lost -> abandon the task
        self.reason = ""
        self.extra: dict = {}

    def beat(self) -> None:
        try:
            r = self.api.call("POST", f"/worker/tasks/{self.task_id}/heartbeat",
                              json={"worker_id": self.worker_id, **self.extra}).json()
            if r.get("cancel"):
                self.reason = "cancelled by user"
                self.cancelled.set()
        except LeaseLost as e:
            self.reason = f"lease lost: {e}"
            self.cancelled.set()
        except RuntimeError as e:              # backend unreachable for a while: keep working, keep trying
            log(f"heartbeat failed: {e}")

    def run(self) -> None:
        while not self.stop_evt.wait(self.every):
            self.beat()
            if self.cancelled.is_set():
                return


# ------------------------------------------------------------------ the worker
class Worker:
    def __init__(self, a):
        self.a = a
        self.api = Api(a.backend_url, a.token)
        self.root = Path(a.root).resolve()
        self.ml_dir = Path(a.ml_dir).resolve() if a.ml_dir else None
        self.pipeline = Path(a.pipeline).resolve() if a.pipeline else (self.ml_dir / "run_pipeline.py")
        self.tmp = self.root.parent / "_worker_tmp"

    # -------------------------------------------------- start-up
    def check(self) -> None:
        if not self.pipeline.is_file():
            raise SystemExit(f"pipeline not found: {self.pipeline} (set --ml-dir to the ML_Work folder)")
        try:
            r = self.api.s.get(self.api.base + "/worker/ping", timeout=20)
        except requests.RequestException as e:
            raise SystemExit(f"backend unreachable at {self.api.base}: {e}")
        if r.status_code == 401:
            raise SystemExit("worker token rejected (401). Use the value of backend/data/.worker_token "
                             "(or BACKEND_WORKER_TOKEN) as W1_WORKER_TOKEN.")
        if r.status_code != 200:
            raise SystemExit(f"/worker/ping -> {r.status_code}: {r.text[:200]}")
        lease = r.json().get("lease_seconds", 300)
        if self.a.heartbeat_seconds >= lease / 2:
            self.a.heartbeat_seconds = max(10, lease / 3)
        log(f"connected to {self.api.base} as '{self.a.worker_id}' (lease {lease}s, "
            f"heartbeat {self.a.heartbeat_seconds:.0f}s)")

    # -------------------------------------------------- main loop
    def loop(self) -> int:
        self.check()
        done, idle_since = 0, time.time()
        while True:
            try:
                r = self.api.call("POST", "/worker/claim", ok=(200, 204),
                                  json={"worker_id": self.a.worker_id, "types": ["reconstruct"]})
            except RuntimeError as e:
                log(f"claim failed: {e}")
                r = None
            if r is not None and r.status_code == 200:
                self.process(r.json())
                done, idle_since = done + 1, time.time()
                if self.a.once or (self.a.max_tasks and done >= self.a.max_tasks):
                    return 0
                continue
            if self.a.once:
                log("no pending task")
                return 0
            if self.a.max_idle_minutes and time.time() - idle_since > self.a.max_idle_minutes * 60:
                log(f"idle for {self.a.max_idle_minutes} min, exiting (saves GPU quota)")
                return 0
            time.sleep(self.a.poll_seconds)

    # -------------------------------------------------- one task
    def process(self, claim: dict) -> None:
        task = claim["task"]
        tid, pid = task["id"], task["project_id"]
        log(f"claimed task {tid} (project {pid}, attempt {task['attempts']}/{task['max_attempts']}, "
            f"args {claim['pipeline_args']})")
        hb = Heartbeat(self.api, tid, self.a.worker_id, self.a.heartbeat_seconds)
        hb.start()
        t0 = time.perf_counter()
        try:
            video = self.tmp / tid / (claim.get("video_filename") or "input.mp4")
            download(self.api, claim["video_url"], video, claim.get("video_size_bytes"), claim.get("video_sha256"))
            log(f"video downloaded ({video.stat().st_size / 2**20:.1f} MB)")

            project = self.root / pid
            if project.exists():                       # leftovers of an earlier attempt must not leak in
                shutil.rmtree(project)
            ok, tail = self.run_pipeline(tid, pid, video, claim["pipeline_args"], hb)
            if hb.cancelled.is_set():
                log(f"task {tid} abandoned: {hb.reason}")
                return
            pipeline_s = time.perf_counter() - t0
            if not ok:
                self.fail(tid, "pipeline failed:\n" + tail, retryable=True)
                return
            missing = [rel for kind, rel, _, req in UPLOADS if req and not (project / rel).exists()]
            if missing:
                self.fail(tid, f"pipeline finished but did not produce {missing}", retryable=False)
                return

            hb.extra = {"stage": "uploading results", "progress": 99}
            hb.beat()
            uploads = self.upload_all(tid, project, hb)
            result = {"worker_id": self.a.worker_id, "host": platform.node(),
                      "pipeline_seconds": round(pipeline_s, 1),
                      "upload_seconds": round(sum(u["seconds"] for u in uploads.values()), 1),
                      "uploads": uploads, "stages": stage_metrics(project / "metrics")}
            self.api.call("POST", f"/worker/tasks/{tid}/complete", json={"worker_id": self.a.worker_id,
                                                                          "result": result})
            log(f"task {tid} done in {(time.perf_counter() - t0) / 60:.1f} min "
                f"(pipeline {pipeline_s / 60:.1f}, upload {result['upload_seconds']:.0f}s)")
        except LeaseLost as e:
            log(f"task {tid} abandoned: lease lost ({e})")
        except Exception as e:                          # noqa: BLE001 - report anything else to the backend
            log(f"task {tid} error: {e!r}")
            self.fail(tid, f"worker error: {e!r}", retryable=True)
        finally:
            hb.stop_evt.set()
            shutil.rmtree(self.tmp / tid, ignore_errors=True)
            if not self.a.keep_workspace:
                shutil.rmtree(self.root / pid, ignore_errors=True)

    def run_pipeline(self, tid, pid, video, extra_args, hb) -> tuple[bool, str]:
        cmd = [sys.executable, str(self.pipeline), "--video", str(video), "--root", str(self.root),
               "--project", pid, "--backend-url", self.api.base, "--task-id", tid, *extra_args]
        env = dict(os.environ, W1_WORKER_TOKEN=self.a.token, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        log("$ " + " ".join(cmd))
        log_dir = self.tmp / tid
        log_dir.mkdir(parents=True, exist_ok=True)
        kw = {"start_new_session": True} if os.name != "nt" else \
             {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        tail: deque[str] = deque(maxlen=60)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                cwd=str(self.pipeline.parent), bufsize=1, errors="replace", **kw)

        def pump():
            for line in proc.stdout:
                if not line.startswith("[progress]"):        # already reported to the backend; not an error
                    tail.append(line.rstrip())
                print(line, end="", flush=True)
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        while proc.poll() is None:
            if hb.cancelled.wait(1.0):
                log(f"stopping pipeline: {hb.reason}")
                kill_tree(proc)
                break
        reader.join(timeout=10)
        return proc.returncode == 0 and not hb.cancelled.is_set(), "\n".join(tail)

    # -------------------------------------------------- uploads
    def upload_all(self, tid: str, project: Path, hb: Heartbeat) -> dict:
        report = {}
        for kind, rel, is_dir, required in UPLOADS:
            if kind == "ply" and not self.a.upload_ply:
                continue
            if kind == "gs" and not self.a.upload_checkpoint:
                continue
            src = project / rel
            if not src.exists():
                if required:
                    raise RuntimeError(f"required artifact missing: {rel}")
                continue
            if hb.cancelled.is_set():
                raise LeaseLost(hb.reason)
            t0 = time.perf_counter()
            path = make_tar(project, rel, self.tmp / tid / f"{kind}.tar") if is_dir else src
            size = path.stat().st_size
            self.upload_file(tid, kind, path)
            dt = time.perf_counter() - t0
            report[kind] = {"mb": round(size / 2**20, 2), "seconds": round(dt, 1),
                            "mb_per_s": round(size / 2**20 / dt, 2) if dt > 0 else None}
            log(f"uploaded {kind:<10} {size / 2**20:8.1f} MB in {dt:6.1f}s")
            if is_dir:
                path.unlink(missing_ok=True)
        return report

    def upload_file(self, tid: str, kind: str, path: Path) -> None:
        size, digest = path.stat().st_size, sha256_file(path)
        for attempt in range(2):                         # second round only after a whole-file checksum failure
            u = self.api.call("POST", f"/worker/tasks/{tid}/uploads", json={
                "worker_id": self.a.worker_id, "kind": kind, "size_bytes": size, "sha256": digest,
                "chunk_size": self.a.chunk_mb * 2**20}).json()
            if u["status"] == "complete":
                return
            if u["received"]:
                log(f"resuming {kind}: {u['received']}/{u['total_chunks']} chunks already on the server")
            with path.open("rb") as f:
                for i in u["missing"]:
                    f.seek(i * u["chunk_size"])
                    data = f.read(u["chunk_size"])
                    self.api.call("PUT", f"/worker/uploads/{u['upload_id']}/chunks/{i}", data=data, headers={
                        "X-Worker-Id": self.a.worker_id, "X-Chunk-Sha256": hashlib.sha256(data).hexdigest(),
                        "Content-Type": "application/octet-stream"})
            try:
                self.api.call("POST", f"/worker/uploads/{u['upload_id']}/complete",
                              json={"worker_id": self.a.worker_id})
                return
            except RuntimeError as e:
                if "sha256 mismatch" not in str(e) or attempt == 1:
                    raise
                log(f"{kind}: checksum mismatch on the server, uploading again")

    def fail(self, tid: str, error: str, retryable: bool) -> None:
        try:
            r = self.api.call("POST", f"/worker/tasks/{tid}/fail",
                              json={"worker_id": self.a.worker_id, "error": error[-15000:], "retryable": retryable})
            log(f"task {tid} reported failed -> now {r.json().get('status')}")
        except (LeaseLost, RuntimeError) as e:
            log(f"could not report failure: {e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend-url", default=os.environ.get("BACKEND_URL", ""), required=not os.environ.get("BACKEND_URL"))
    ap.add_argument("--token", default=os.environ.get("W1_WORKER_TOKEN", ""), help="default: env W1_WORKER_TOKEN")
    ap.add_argument("--ml-dir", default=os.environ.get("ML_DIR", ""), help="Person A's ML_Work folder")
    ap.add_argument("--pipeline", default="", help="alternative pipeline script with run_pipeline.py's CLI (tests)")
    ap.add_argument("--root", default=os.environ.get("W1_ROOT", "/kaggle/working/workspace"))
    ap.add_argument("--worker-id", default=f"{socket.gethostname()}-{os.getpid()}")
    ap.add_argument("--poll-seconds", type=float, default=10)
    ap.add_argument("--heartbeat-seconds", type=float, default=60)
    ap.add_argument("--chunk-mb", type=int, default=32)
    ap.add_argument("--upload-ply", action="store_true", help="also upload export/scene.ply (~250 MB)")
    ap.add_argument("--upload-checkpoint", action="store_true",
                    help="also upload gs/ (trainer checkpoints) for the Week 2 fine-tune")
    ap.add_argument("--once", action="store_true", help="process at most one task, then exit")
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--max-idle-minutes", type=float, default=0, help="exit after this long without work (0 = never)")
    ap.add_argument("--keep-workspace", action="store_true", help="keep the local project folder after upload")
    a = ap.parse_args(argv)
    if not a.token:
        ap.error("no worker token: set W1_WORKER_TOKEN or pass --token")
    if not a.ml_dir and not a.pipeline:
        ap.error("--ml-dir is required")
    return Worker(a).loop()


if __name__ == "__main__":
    sys.exit(main())
