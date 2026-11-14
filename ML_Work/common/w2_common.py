"""Shared helpers of the Week 2 stages (w2_segment / w2_track / w2_rebuild). Stage CLI contract = Person B's worker:
    --project-dir <root>/<project_id> --edit-id E [--backend-url URL --task-id ID]   (appended by the worker)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common.colmap_io import read_model
from common.config import Config, read_json
from common.lift import Cam
from common.masks import edit_dir
from common.progress import Progress


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", type=Path, required=True, help="<root>/<project_id> (worker passes this)")
    ap.add_argument("--edit-id", required=True)
    ap.add_argument("--backend-url", default="")
    ap.add_argument("--task-id", default="")
    ap.add_argument("--sam-model", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--sam-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    return ap


def cfg_from_dir(project_dir: Path) -> Config:
    pd = Path(project_dir).resolve()
    return Config(root=str(pd.parent), project_id=pd.name)


def edit_progress(cfg: Config, edit_id: str, args) -> Progress:
    return Progress(edit_dir(cfg.project, edit_id) / "metrics", args.backend_url, args.task_id)


def load_inputs(cfg: Config) -> dict:
    """week2/inputs.json (written by stage 6) or a best-effort reconstruction when only the Week 1 files exist."""
    d = read_json(cfg.p("week2", "inputs.json"))
    if d:
        return d
    idx = read_json(cfg.frames_index, {})
    return {"images": {"same_pixels": True, "frames_size_wh": idx.get("image_size_wh"), "scene_size_wh": None},
            "depth": {"kind": "none", "dir": None}, "heldout": {"every": 8, "names": []}, "checkpoint": None,
            "_inferred": True}


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def clicks_to_points(clicks: list[dict]) -> list[tuple[float, float, int]]:
    return [(float(c["x"]), float(c["y"]), 1 if c.get("positive", True) else 0) for c in clicks]


def load_cams(cfg: Config) -> dict[str, Cam]:
    """name -> Cam (scene/images pixel space) for every registered frame of scene/sparse/0."""
    from common import gs_space
    model = read_model(cfg.scene_dir / "sparse" / "0")
    splat_poses = gs_space.load_poses(cfg.p("week2", gs_space.FILE))     # None until stages/w2_calibrate.py ran
    out = {}
    for im in model.images.values():
        c = model.cameras[im.camera_id]
        R, t = im.R, im.tvec
        if splat_poses is not None and im.name in splat_poses:           # the splat depth was rendered with these
            V = splat_poses[im.name]
            R, t = V[:3, :3], V[:3, 3]
        out[im.name] = Cam(c.K(), R, t, c.width, c.height)
    return out


def load_depth(cfg: Config, inputs: dict, name: str) -> np.ndarray | None:
    d = (inputs.get("depth") or {}).get("dir")
    if not d:
        return None
    p = cfg.p(d, Path(name).stem + ".npy")
    return np.load(p).astype(np.float32) if p.exists() else None


def space_maps(cfg: Config, inputs: dict):
    """(frames->scene map, scene->frames map) when the two image spaces differ, else (None, None)."""
    im = inputs.get("images", {})
    if im.get("same_pixels", True):
        return None, None
    from common import undist
    raw = im.get("raw_camera")
    cam = (inputs.get("poses") or {}).get("camera")
    if not raw or not cam:
        raise RuntimeError("scene/images differs from frames/images but week2/inputs.json has no raw_camera - re-run "
                           "stage 2b + stage 6 (or make masks on scene/images)")
    return undist.frames_to_scene_map(raw, cam), undist.scene_to_frames_map(raw, cam)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
