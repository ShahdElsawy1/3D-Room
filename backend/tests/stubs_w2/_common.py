"""Shared bits of the Week 2 stage stand-ins (tests only; the real stages are Person A's / C's)."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=Path, required=True)
    ap.add_argument("--edit-id", required=True)
    ap.add_argument("--backend-url", default="")
    ap.add_argument("--task-id", default="")
    return ap


def progress(a, stage, pct):
    if not (a.backend_url and a.task_id):
        return
    import requests
    tok = os.environ.get("W1_WORKER_TOKEN")
    requests.post(f"{a.backend_url}/tasks/{a.task_id}/progress", timeout=5,
                  json={"status": "running", "stage": stage, "progress": pct},
                  headers={"Authorization": f"Bearer {tok}"} if tok else None)


def frames(a):
    return sorted(p.name for p in (a.project_dir / "frames" / "images").iterdir() if p.suffix in (".jpg", ".png"))


def mask_dir(a):
    d = a.project_dir / "edits" / a.edit_id / "masks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ellipse_mask(frame: Path, cx: int, cy: int, rx=70, ry=45) -> Image.Image:
    w, h = Image.open(frame).size
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    return Image.fromarray((m * 255).astype(np.uint8))


def read_index(a):
    f = mask_dir(a) / "index.json"
    return json.loads(f.read_text()) if f.exists() else {"ref_frame": None, "method": "stub", "frames": []}


def write_index(a, idx):
    by = {f["name"]: f for f in idx["frames"]}
    idx["frames"] = [by.get(n, {"name": n, "present": False, "source": None, "score": 0.0, "area_px": 0})
                     for n in frames(a)]
    (mask_dir(a) / "index.json").write_text(json.dumps(idx, indent=2))
