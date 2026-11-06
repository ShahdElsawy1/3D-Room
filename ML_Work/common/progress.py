"""Real stage progress for the backend's progress bar (no fake timers).

Contract (matches the PDF, section 4.4):
    POST {backend}/tasks/{task_id}/progress
    body: {"status": "running|done|failed", "stage": "...", "progress": 0-100, "error": null}

Overall progress is split across the four user-visible stages:
    extracting frames   0 - 15
    estimating poses   15 - 45
    training           45 - 90
    exporting          90 - 100

Everything is also appended to <project>/metrics/progress.jsonl so a run can be
inspected (or replayed) when no backend is reachable. Network errors never crash a stage.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

STAGE_RANGES = {
    "extracting frames": (0, 15),
    "estimating poses": (15, 45),
    "training": (45, 90),
    "exporting": (90, 100),
}


class Progress:
    def __init__(self, metrics_dir, backend_url: str = "", task_id: str = ""):
        self.log = Path(metrics_dir) / "progress.jsonl"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.url = backend_url.rstrip("/") if backend_url else ""
        self.task_id = task_id
        self._last_post = 0.0

    def report(self, stage: str, frac: float, status: str = "running", error: str | None = None,
               force: bool = False):
        lo, hi = STAGE_RANGES.get(stage, (0, 100))
        frac = max(0.0, min(1.0, float(frac)))
        pct = round(lo + (hi - lo) * frac, 1)
        rec = {"t": time.time(), "status": status, "stage": stage, "progress": pct, "error": error}
        with self.log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[progress] {stage}: {pct}%", flush=True)
        if self.url and self.task_id:
            # throttle to <= 2 posts / second unless forced (stage boundaries, errors)
            now = time.time()
            if force or status != "running" or now - self._last_post > 0.5:
                self._last_post = now
                try:
                    import requests
                    # optional worker token (backend exposed via Cloudflare); unset -> no header, as before
                    token = os.environ.get("W1_WORKER_TOKEN")
                    headers = {"Authorization": f"Bearer {token}"} if token else None
                    requests.post(f"{self.url}/tasks/{self.task_id}/progress", json=rec, headers=headers, timeout=5)
                except Exception as e:  # backend down / tunnel dropped: keep working
                    print(f"[progress] backend unreachable ({e.__class__.__name__}); continuing", flush=True)

    def start(self, stage): self.report(stage, 0.0, force=True)
    def finish(self, stage): self.report(stage, 1.0, force=True)
    def fail(self, stage, err): self.report(stage, 0.0, status="failed", error=str(err), force=True)
    def done(self): self.report("exporting", 1.0, status="done", force=True)
