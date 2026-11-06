#!/usr/bin/env python3
"""Diffusion edit stage (Week 2, Person B). Same style as Person A's stage scripts: one process per stage, so GPU
memory is released when it ends, files in the project folder, real progress posted to the backend.

  # 1. reference frame, several seeds -> candidates the user picks from (spec §4.3 "first edited frame")
  python run_edit.py preview --project-dir <root>/<pid> --edit-id <eid> --prompt "dark green leather sofa" --seeds 0,1,2
  # 2. every other frame where the object is present, conditioned on the chosen candidate
  python run_edit.py frames  --project-dir <root>/<pid> --edit-id <eid> --prompt "..." --chosen-seed 1

Inputs (contract in docs/W2_INTERFACES_PROPOSAL.md):
  frames/images/<name>                         Week 1 frames (unchanged)
  edits/<eid>/masks/<stem>.png + index.json    Person A: 0/255 masks, frame size, "present" flag per frame
Outputs:
  edits/<eid>/candidates/seed_<n>.png (+ _crop.png) + index.json
  edits/<eid>/edited/<stem>.png + index.json   composited frames, same size as the originals, PNG (lossless)
  edits/<eid>/edit_report.json                 model, dtype, per-frame seconds / peak VRAM / leak / colour drift
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):  # run as a script on Kaggle: python backend/editing/run_edit.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "editing"

from editing import composite as C  # noqa: E402
from editing import geometry as G  # noqa: E402
from editing import metrics as M  # noqa: E402
from editing.editors import Editor, make_editor  # noqa: E402


# ---------------------------------------------------------------------------------------------------- progress
class Progress:
    """POST {backend}/tasks/{id}/progress like Week 1 (same body, W1_WORKER_TOKEN header). Never fails the stage."""

    def __init__(self, log_path: Path, backend_url: str = "", task_id: str = ""):
        self.log = log_path
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.url, self.task_id, self._last = backend_url.rstrip("/"), task_id, 0.0

    def report(self, stage: str, pct: float, status: str = "running", error: str | None = None, force=False):
        rec = {"t": time.time(), "status": status, "stage": stage, "progress": round(pct, 1), "error": error}
        with self.log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[progress] {stage}: {rec['progress']}%", flush=True)
        now = time.time()
        if self.url and self.task_id and (force or status != "running" or now - self._last > 0.5):
            self._last = now
            try:
                import requests
                tok = os.environ.get("W1_WORKER_TOKEN")
                requests.post(f"{self.url}/tasks/{self.task_id}/progress", json=rec, timeout=5,
                              headers={"Authorization": f"Bearer {tok}"} if tok else None)
            except Exception as e:
                print(f"[progress] backend unreachable ({e.__class__.__name__}); continuing", flush=True)


# ---------------------------------------------------------------------------------------------------- inputs
class InputError(ValueError):
    pass


@dataclass
class Paths:
    project: Path
    edit: Path

    @property
    def images(self): return self.project / "frames" / "images"
    @property
    def masks(self): return self.edit / "masks"
    @property
    def candidates(self): return self.edit / "candidates"
    @property
    def edited(self): return self.edit / "edited"
    @property
    def report(self): return self.edit / "edit_report.json"

    def mask(self, name: str) -> Path: return self.masks / (Path(name).stem + ".png")


def read_masks_index(p: Paths) -> dict:
    f = p.masks / "index.json"
    if not f.exists():
        raise InputError(f"{f} missing (Person A's tracking output)")
    idx = json.loads(f.read_text())
    for fr in idx.get("frames", []):
        if not (p.images / fr["name"]).exists():
            raise InputError(f"masks/index.json lists {fr['name']}, which is not in frames/images")
        if fr.get("present") and not p.mask(fr["name"]).exists():
            raise InputError(f"{fr['name']} is present but {p.mask(fr['name']).name} is missing")
    if not any(fr.get("present") for fr in idx.get("frames", [])):
        raise InputError("no frame is marked present")
    return idx


def present_frames(idx: dict) -> list[str]:
    """Present frames in temporal order (order of frames/images names, which Week 1 numbers in order)."""
    return sorted(fr["name"] for fr in idx["frames"] if fr.get("present"))


def pick_reference(idx: dict, requested: str | None) -> str:
    """Spec: the frame where the object is large and clearly visible. Default: the user's click frame, else the
    present frame with the largest mask."""
    present = present_frames(idx)
    for name in (requested, idx.get("ref_frame")):
        if name:
            if name not in present:
                raise InputError(f"reference frame {name} is not a present frame")
            return name
    return max((fr for fr in idx["frames"] if fr.get("present")), key=lambda fr: fr.get("area_px", 0))["name"]


def edit_order(present: list[str], ref: str) -> list[tuple[str, str]]:
    """(frame, neighbour towards the reference): ref+1, ref+2, ... then ref-1, ref-2, ... so 'previous edited
    frame' conditioning always uses a frame that was edited just before and is adjacent in time."""
    i = present.index(ref)
    fwd = [(present[j], present[j - 1]) for j in range(i + 1, len(present))]
    bwd = [(present[j], present[j + 1]) for j in range(i - 1, -1, -1)]
    return fwd + bwd


# ---------------------------------------------------------------------------------------------------- one frame
@dataclass
class Prep:
    name: str
    orig: np.ndarray
    mask: np.ndarray
    region: np.ndarray  # dilated mask: the only pixels allowed to change
    box: G.Box
    alpha: np.ndarray
    size: tuple[int, int]
    model_in: Image.Image


def prepare(p: Paths, name: str, a) -> Prep:
    orig = G.load_image(p.images / name)
    mask = G.load_mask(p.mask(name), (orig.shape[1], orig.shape[0]))
    region = G.dilate(mask, a.dilate)
    box = G.crop_box(region, a.margin)
    size = G.model_size(box.w, box.h, a.long_side)
    return Prep(name, orig, mask, region, box, C.alpha_map(region, a.feather), size,
                G.to_model(orig[box.slices], size))


def finish(prep: Prep, raw: Image.Image) -> tuple[np.ndarray, np.ndarray, dict]:
    raw_crop = G.from_model(raw, prep.box)
    out = C.composite(prep.orig, raw_crop, prep.box, prep.alpha)
    C.assert_unchanged_outside(prep.orig, out, prep.region, prep.name)
    rb = prep.region[prep.box.slices]
    m = {"raw_leak_psnr": M.json_num(M.raw_leak_psnr(prep.orig[prep.box.slices], raw_crop, rb)),
         "changed_outside_px": C.changed_outside(prep.orig, out, prep.region),
         "box": prep.box.as_list(), "model_size": list(prep.size),
         "lab_edited": M.mean_lab(out, prep.mask)}
    return out, out[prep.box.slices], m


def save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    Image.fromarray(arr).save(tmp, compress_level=4)
    tmp.replace(path)


def update_report(p: Paths, section: str, data: dict) -> None:
    rep = json.loads(p.report.read_text()) if p.report.exists() else {}
    rep[section] = data
    p.report.parent.mkdir(parents=True, exist_ok=True)
    p.report.write_text(json.dumps(rep, indent=2))


def common_info(a, editor: Editor) -> dict:
    return {**editor.info(), "prompt": a.prompt, "long_side": a.long_side, "margin": a.margin,
            "dilate_px": a.dilate, "feather_px": a.feather}


def summary(frames: list[dict]) -> dict:
    sec = [f["seconds"] for f in frames]
    vram: dict = {}
    for f in frames:
        for k, v in f.get("peak_vram_mb", {}).items():
            vram[k] = max(vram.get(k, 0), v)
    leaks = [f["raw_leak_psnr"] for f in frames if isinstance(f.get("raw_leak_psnr"), (int, float))]
    de = [f["delta_e_vs_reference"] for f in frames if f.get("delta_e_vs_reference") is not None]
    return {"frames": len(frames), "total_seconds": round(sum(sec), 1),
            "seconds_per_frame": round(sum(sec) / len(sec), 2) if sec else None,
            "peak_vram_mb": vram, "degenerate_frames": sum(bool(f.get("degenerate")) for f in frames),
            "mean_raw_leak_psnr": round(float(np.mean(leaks)), 2) if leaks else None,
            "mean_delta_e_vs_reference": round(float(np.mean(de)), 2) if de else None,
            "max_delta_e_vs_reference": max(de) if de else None}


# ---------------------------------------------------------------------------------------------------- modes
def cmd_preview(a, p: Paths, editor: Editor, prog: Progress) -> dict:
    idx = read_masks_index(p)
    ref = pick_reference(idx, a.ref_frame)
    seeds = [int(s) for s in a.seeds.split(",")]
    prep = prepare(p, ref, a)
    stage = "editing reference frame"
    out_seeds = []
    for i, seed in enumerate(seeds):
        prog.report(stage, 100 * i / len(seeds), force=i == 0)
        raw, st = editor.edit([prep.model_in], a.prompt, seed, prep.size)
        full, crop, m = finish(prep, raw)
        save_png(full, p.candidates / f"seed_{seed}.png")
        save_png(crop, p.candidates / f"seed_{seed}_crop.png")
        out_seeds.append({"seed": seed, "file": f"candidates/seed_{seed}.png", "crop": f"candidates/seed_{seed}_crop.png",
                          "seconds": st.seconds, "peak_vram_mb": st.peak_vram_mb, "degenerate": st.degenerate, **m})
    cand = {"ref_frame": ref, "prompt": a.prompt, "box": prep.box.as_list(), "seeds": out_seeds}
    (p.candidates / "index.json").write_text(json.dumps(cand, indent=2))
    update_report(p, "preview", {**common_info(a, editor), "ref_frame": ref, "candidates": out_seeds,
                                 "summary": summary(out_seeds)})
    prog.report(stage, 100, force=True)
    return cand


def cmd_frames(a, p: Paths, editor: Editor, prog: Progress) -> dict:
    idx = read_masks_index(p)
    cf = p.candidates / "index.json"
    if not cf.exists():
        raise InputError("no candidates: run 'preview' first")
    cand = json.loads(cf.read_text())
    chosen = next((s for s in cand["seeds"] if s["seed"] == a.chosen_seed), None)
    if chosen is None:
        raise InputError(f"seed {a.chosen_seed} is not among the candidates {[s['seed'] for s in cand['seeds']]}")
    ref = cand["ref_frame"]
    ref_prep = prepare(p, ref, a)
    ref_crop = np.asarray(Image.open(p.edit / chosen["crop"]).convert("RGB"))
    ref_img = G.to_model(ref_crop, ref_prep.size)
    ref_lab = M.mean_lab(np.asarray(Image.open(p.edit / chosen["file"]).convert("RGB")), ref_prep.mask)
    seed = a.seed if a.seed is not None else a.chosen_seed  # spec idea: same seed + prompt for every frame

    # the reference frame's edit is the chosen candidate itself
    save_png(np.asarray(Image.open(p.edit / chosen["file"]).convert("RGB")), p.edited / (Path(ref).stem + ".png"))
    excluded = set(filter(None, a.exclude.split(","))) - {ref}   # the user removed these in the mask grid
    present = [n for n in present_frames(idx) if n not in excluded]
    order = edit_order(present, ref)
    if a.only:
        wanted = set(a.only.split(","))
        order = [o for o in order if o[0] in wanted]
    skipped = []
    if a.keyframe_every > 1:  # spec idea: edit keyframes only; the rest must be dropped from the rebuild
        keep = set(present[::a.keyframe_every]) | {ref}
        skipped = [n for n, _ in order if n not in keep]
        order = [o for o in order if o[0] in keep]

    crops: dict[str, Image.Image] = {ref: ref_img}
    frames = []
    n = len(order)
    for i, (name, neighbour) in enumerate(order):
        prog.report(f"editing frame {i + 1} of {n}", 100 * i / max(n, 1), force=i == 0)
        prep = prepare(p, name, a)
        images = [prep.model_in, ref_img]
        if a.with_previous and neighbour != ref and neighbour in crops:
            images.append(crops[neighbour])  # watch for error accumulation (spec §4.3)
        raw, st = editor.edit(images, a.prompt, seed, prep.size)
        full, crop, m = finish(prep, raw)
        save_png(full, p.edited / (Path(name).stem + ".png"))
        crops[name] = G.to_model(crop, prep.size)
        frames.append({"name": name, "seconds": st.seconds, "peak_vram_mb": st.peak_vram_mb,
                       "degenerate": st.degenerate, "n_condition_images": len(images),
                       "delta_e_vs_reference": M.delta_e(m["lab_edited"], ref_lab), **m})
    edited = [ref] + [f["name"] for f in frames]
    out = {"ref_frame": ref, "chosen_seed": a.chosen_seed, "seed": seed, "with_previous": a.with_previous,
           "edited": [{"frame": nm, "file": f"edited/{Path(nm).stem}.png"} for nm in edited],
           "skipped_present_frames": skipped, "excluded_by_user": sorted(excluded & set(present_frames(idx)))}
    (p.edited / "index.json").write_text(json.dumps(out, indent=2))
    update_report(p, "frames", {**common_info(a, editor), **{k: out[k] for k in ("ref_frame", "chosen_seed", "seed",
                  "with_previous", "skipped_present_frames")}, "frames": frames, "summary": summary(frames)})
    prog.report(f"editing frame {n} of {n}", 100, force=True)
    return out


# ---------------------------------------------------------------------------------------------------- cli
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["preview", "frames"])
    ap.add_argument("--project-dir", required=True, type=Path)
    ap.add_argument("--edit-id", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default="klein", choices=["klein", "qwen-turbo", "qwen", "fake"])
    ap.add_argument("--precision", default="bf16", help="bf16 (A) | bf16-fp32 (B) | fp8-bf16 (D) | fp16 (C, needs --allow-fp16)")
    ap.add_argument("--allow-fp16", action="store_true")
    ap.add_argument("--placement", default="split", choices=["cuda", "split", "offload"])
    ap.add_argument("--steps", type=int, default=None, help="override the model's default number of steps")
    ap.add_argument("--upsampler", action="store_true", help="Klein: keep the 9B prompt upsampler (does not fit a T4)")
    ap.add_argument("--long-side", type=int, default=768, help="crop is resized so its long side is about this")
    ap.add_argument("--margin", type=float, default=0.15, help="context around the mask, fraction of its size")
    ap.add_argument("--dilate", type=int, default=8, help="px the mask is grown; nothing outside it can change")
    ap.add_argument("--feather", type=int, default=6, help="px of soft blend, inside the dilated mask")
    # preview
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ref-frame", default=None)
    # frames
    ap.add_argument("--chosen-seed", type=int)
    ap.add_argument("--seed", type=int, default=None, help="seed for the other frames (default: the chosen seed)")
    ap.add_argument("--with-previous", action="store_true", help="also condition on the previous edited frame")
    ap.add_argument("--keyframe-every", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-separated frame names to (re)edit")
    ap.add_argument("--exclude", default="", help="comma-separated frame names the user removed (not edited)")
    ap.add_argument("--backend-url", default="")
    ap.add_argument("--task-id", default="")
    a = ap.parse_args(argv)
    if a.mode == "frames" and a.chosen_seed is None:
        ap.error("frames needs --chosen-seed")
    if a.feather > a.dilate:
        ap.error("--feather must be <= --dilate so the object itself is fully replaced")
    return a


def main(argv=None, editor: Editor | None = None) -> int:
    a = parse(argv)
    p = Paths(a.project_dir, a.project_dir / "edits" / a.edit_id)
    prog = Progress(p.edit / "progress.jsonl", a.backend_url, a.task_id)
    stage = "editing reference frame" if a.mode == "preview" else "editing frames"
    try:
        prog.report(stage if a.mode == "preview" else "loading model", 0, force=True)
        editor = editor or make_editor(a.model, a.precision, a.placement, a.allow_fp16, a.steps, a.upsampler)
        (cmd_preview if a.mode == "preview" else cmd_frames)(a, p, editor, prog)
        return 0
    except Exception as e:
        traceback.print_exc()
        prog.report(stage, 0, status="failed", error=f"{e.__class__.__name__}: {e}", force=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
