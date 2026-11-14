#!/usr/bin/env python3
"""Pack Person A's complete work for Person B (backend + diffusion) and Person C (UI) - generated from the real files.

    python tools/make_handoff.py --project-dir $W1_ROOT/demo --out /kaggle/working/handoff_demo.zip
    python tools/make_handoff.py --project-dir ... --tables-only          # just (re)write the Markdown tables

The zip root is the project folder (`<project_id>/...`): unzip it into backend/data/workspace/ and the backend sees
exactly the layout the worker uses. Contents:
    Week 1 : frames/, scene/ (sparse + images), export/ (scene.splat, scene.ply, cameras.json, scene_meta.json),
             week2/ (inputs.json, depth, sparse_depth), gs/latest.json (+ checkpoint unless --no-checkpoint),
             manifest.json, metrics/*.json, poses/vggt depth when VGGT is the chosen method
    Week 2 : edits/<E>/{masks, gt, track_eval, metrics, rebuilds/*, export, edited, preview*.png} per edit
    HANDOFF/ README_FOR_B.md, README_FOR_C.md, SUMMARY.md  (numbers are read from the metrics files, not typed)
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import read_json  # noqa: E402
from common.w2_common import cfg_from_dir  # noqa: E402


def f(v, nd=2, unit=""):
    return "n/a" if v is None else (f"{v:.{nd}f}{unit}" if isinstance(v, float) else f"{v}{unit}")


# ----------------------------------------------------------------------------- tables from measurements
def week1_summary(cfg) -> dict:
    ch = read_json(cfg.scene_dir / "chosen.json", {})
    meth = ch.get("method")
    mt = lambda k: read_json(cfg.metrics_dir / f"{k}.json", {}) or {}   # noqa: E731
    tr, ex, fr = mt("s4_train_3dgs"), mt("s5_export_splat"), mt("s1_extract_frames")
    gm = read_json(cfg.poses_dir("glomap") / "glomap_meta.json", {}) or {}
    vg = mt("s2a_poses_vggt")
    pc = read_json(cfg.metrics_dir / "pose_comparison.json", {}) or {}
    return {"method": meth, "train": tr, "export": ex, "frames": fr, "glomap": gm, "vggt": vg, "compare": pc,
            "inputs": read_json(cfg.p("week2", "inputs.json"), {}) or {}}


def rebuild_comparison_md(E: Path) -> str | None:
    rows = []
    for mode in ("scratch", "finetune"):
        r = read_json(E / "rebuilds" / mode / "report.json")
        if r:
            ho, h0 = r.get("heldout_edited") or {}, r.get("heldout_original") or {}
            rows.append(f"| {mode} | {r['iters']} | {f(r['seconds']['train'], 0, ' s')} | {f(r['seconds']['total'], 0, ' s')} | "
                        f"{r.get('num_gaussians')} | {f(ho.get('psnr'))} | {f(ho.get('ssim'), 3)} | "
                        f"{f(h0.get('psnr'))} |")
    if not rows:
        return None
    return ("| mode | iterations | training time | total time | Gaussians | held-out PSNR (edited scene) | SSIM | "
            "PSNR of the ORIGINAL splat on the original frames |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))


def edits(cfg) -> list[Path]:
    d = cfg.p("edits")
    return sorted(p for p in d.iterdir() if p.is_dir()) if d.exists() else []


def summary_md(cfg) -> str:
    s = week1_summary(cfg)
    tr, gm, vg, inp = s["train"], s["glomap"], s["vggt"], s["inputs"]
    ho = tr.get("heldout") or {}
    L = [f"# Person A - hand-off summary (project `{cfg.project_id}`)", "",
         "All numbers below are read from `metrics/*.json` and `edits/*/metrics/*.json` at packing time.", "",
         "## Week 1 - video -> splat",
         f"- Chosen pose method: **{s['method']}**   (both methods are compared in `metrics/pose_comparison.md`)",
         f"- Frames: {inp.get('total_frames', 'n/a')}, registered: {inp.get('registered_frames', 'n/a')}",
         f"- GLOMAP: backend `{gm.get('backend')}`, mapper mode `{gm.get('mapper_mode')}`, registered "
         f"{gm.get('registered_frames')}/{gm.get('total_frames')}, {gm.get('points3D')} points, seconds {gm.get('seconds')}",
         f"- VGGT: {vg.get('seconds') and round(vg['seconds'])} s, peak VRAM {vg.get('peak_vram_mb')} MiB",
         f"- Training: {tr.get('trainer')} {tr.get('iters')} iterations, {tr.get('num_gaussians')} Gaussians, "
         f"{f(tr.get('train_wall_seconds'), 0, ' s')}, peak VRAM {tr.get('peak_vram_mb')} MiB",
         f"- Held-out quality: PSNR {f(ho.get('psnr'))}, SSIM {f(ho.get('ssim'), 3)}, LPIPS {f(ho.get('lpips'), 3)}",
         f"- Splat: {s['export'].get('num_gaussians')} Gaussians exported, {f(s['export'].get('splat_mb'), 1, ' MB')} .splat", ""]
    for E in edits(cfg):
        L += [f"## Week 2 - edit `{E.name}`"]
        idx = read_json(E / "masks" / "index.json")
        if idx:
            fr = idx["frames"]
            by = {}
            for x in fr:
                if x["present"]:
                    by[x["source"]] = by.get(x["source"], 0) + 1
            L += [f"- Masks: reference `{idx.get('ref_frame')}`, method `{idx.get('method')}`, "
                  f"{sum(x['present'] for x in fr)}/{len(fr)} frames contain the object, by source {by}"]
        for key, label in (("w2_segment", "segment"), ("w2_track", "track")):
            m = read_json(E / "metrics" / f"{key}.json")
            if m:
                L += [f"- `{label}`: {f(m.get('seconds'), 1, ' s')}, peak VRAM {m.get('peak_vram_mb')} MiB"]
        tr_md = E / "metrics" / "tracking_report.md"
        if tr_md.exists():
            L += ["", tr_md.read_text().replace("# Tracking evaluation", "### Tracking evaluation"), ""]
        cmp_ = rebuild_comparison_md(E)
        if cmp_:
            L += ["", "### Rebuild: fine-tune vs from scratch", "", cmp_, ""]
            r = read_json(E / "rebuilds" / "scratch" / "report.json") or read_json(E / "rebuilds" / "finetune" / "report.json")
            if r:
                jr = r.get("jpeg_reencode", {})
                L += [f"JPEG re-encode of edited frames (q100) changed pixels outside the mask by at most "
                      f"{jr.get('outside_mask_max_abs_diff')} levels (mean {jr.get('outside_mask_mean_abs_diff')}).", ""]
        L.append("")
    L += ["## Known limits", "- SAM 2 / lifting / fine-tune were validated on the team's video only; accuracy numbers above are on "
          "the hand-labelled frames listed in the tracking report.",
          "- Splat depth for lifting is rendered from the trained splat; if it is missing (no checkpoint) only SAM 2 "
          "propagation runs.", ""]
    return "\n".join(L)


README_B = """# For Person B (backend + diffusion)

## What is in the zip
Unzip it into `backend/data/workspace/` - the root folder is the project id. Layout = the worker's layout:
`frames/`, `scene/` (sparse/0 + images), `export/`, `week2/`, `gs/latest.json` (+ checkpoint), `metrics/`, `manifest.json`,
and `edits/<edit_id>/` for the edits A tested. Every path inside the JSON files is relative to the project folder.

## Replace `ML_Work/` with the new copy (not only `progress.py`)
Added since your copy: GLOMAP via pycolmap 4 (`--backend pycolmap`), project-relative paths in all JSON, the
`week2/` stage, and the three Week 2 stages. `common/progress.py` keeps your `W1_WORKER_TOKEN` header (identical logic).

## Week 2 stages (CLI exactly as your worker calls them: `--project-dir <root>/<id> --edit-id E` + `--backend-url --task-id`)
| task type | script | extra args | reads | writes |
|---|---|---|---|---|
| `segment` | `ML_Work/stages/w2_segment.py` | `--frame F --clicks JSON` or `--add-frames JSON` | `frames/` | `edits/E/masks/*.png`, `index.json` |
| `track` | `ML_Work/stages/w2_track.py` | `[--tracker both\\|sam2\\|lift]` | `frames/`, `scene/sparse`, `week2/` (depth), `masks/` | `masks/`, `metrics/track_debug.json` |
| `rebuild` | `ML_Work/stages/w2_rebuild.py` | `--mode finetune\\|scratch` | `scene/`, `edited/`, `masks/`, `week2/inputs.json`, `gs/` checkpoint (finetune) | `edits/E/export/{scene.splat,cameras.json,scene_meta.json,scene.ply}`, `metrics/s4_rebuild.json` |
Progress stage names: `segmenting`, `tracking`, `training` -> `exporting` (rebuild). GPU 1 for all three.
Setup on the worker: `bash scripts/setup_kaggle.sh w2` (SAM 2.1 + gsplat; no VGGT/pycolmap needed for Week 2).

## Files the worker must upload after a Week 1 run (so Week 2 can start from the backend)
`week2/inputs.json` lists them under `required_files` (frames, scene, export, week2/depth + sparse_depth, gs/latest.json and the
checkpoint). Upload the `week2/` folder and the checkpoint (`gs/.../ckpts/*.pt`) - finetune needs it.

## Masks contract (A -> B)
`edits/E/masks/<stem>.png`: 8-bit, 0/255, exactly the size of `frames/images/<name>`; `index.json` has one entry per frame
(`present`, `source`: click | user | propagated | lifted, `score`, `area_px`). `scene/images` may differ from `frames/images` ONLY
when GLOMAP undistorted (see `week2/inputs.json -> images.same_pixels`; the rebuild converts for you).

## Edited frames (B -> A)
Write `edits/E/edited/<stem>.png` for each edited frame and `edits/E/edited/index.json`:
`{"edited": [{"name": "000012.jpg", "file": "edited/000012.png"}, ...]}`.
**Decision (answers your PNG-under-.jpg question):** the rebuild writes a COPY of the training images in `edits/E/scene/images/`
where edited frames are JPEG quality 100 / 4:4:4 under the original name (gsplat reads by name). Measured change outside the mask is
in `SUMMARY.md` (`JPEG re-encode ... at most N levels`). Un-edited frames are symlinks; the original `scene/` is never modified.

## Numbers
See `SUMMARY.md` (Week 1 quality/timings, tracking evaluation, fine-tune vs scratch).
"""

README_C = """# For Person C (viewer + edit UI)

## Viewer data (unchanged from Week 1)
`export/scene.splat` (32 bytes per Gaussian, world is y-down), `export/cameras.json` (every training camera: position, rotation,
intrinsics - use it to snap the viewer to a frame or to pick in 3D), `export/scene_meta.json`.

## Click -> mask
Click coordinates are pixels of the frame image the user sees (`frames/images/<name>`, not normalised, origin top-left).
`positive: false` = "not this" click. Refining = send the full click list again. Latency per click on the queue is the model
load (~seconds) + prediction; numbers in `SUMMARY.md`.

## Frame grid (from `edits/E/masks/index.json`, served by B)
Per frame: `present` (object in the frame), `source` (`click` reference, `propagated` SAM 2, `lifted` 3D reprojection,
`user` marked by the user), `score` (0-1 confidence; show low scores as "check this"), `area_px`. Frames the tracker missed can be
added by the user with new clicks (`PATCH .../frames add`); wrong ones are removed (present=false).

## Before / after
Original splat = `export/scene.splat`. Rebuilt = `edits/E/export/scene.splat` (same cameras). Two rebuild modes exist; the
comparison table (time and quality) is in `SUMMARY.md` - show the time in the UI so users can choose.
"""


# ----------------------------------------------------------------------------- packing
def pack(cfg, out: Path, include_ckpt: bool = True) -> list[str]:
    P = cfg.project
    inp = read_json(cfg.p("week2", "inputs.json"), {}) or {}
    paths: list[Path] = []

    def add(rel: str):
        p = P / rel
        if p.is_dir():
            paths.extend(q for q in p.rglob("*") if q.is_file() or q.is_symlink())
        elif p.exists():
            paths.append(p)

    for rel in inp.get("required_files", []):
        add(rel)
    for rel in ("manifest.json", "week2", "metrics", "export", "scene", "frames", "gs/latest.json", "scene/chosen.json"):
        add(rel)
    if not include_ckpt:
        paths = [p for p in paths if p.suffix != ".pt"]
    for E in edits(cfg):
        for sub in ("masks", "gt", "track_eval", "metrics", "rebuilds", "export", "edited"):
            add(f"edits/{E.name}/{sub}")
        for png in E.glob("preview*.png"):
            paths.append(png)
    seen, files = set(), []
    for p in paths:
        key = p.relative_to(P).as_posix()
        if key not in seen and p.is_file():                  # symlinks are followed: the zip holds real files
            seen.add(key)
            files.append((key, p))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for key, p in files:
            z.write(p, f"{cfg.project_id}/{key}")
        z.writestr(f"{cfg.project_id}/HANDOFF/SUMMARY.md", summary_md(cfg))
        z.writestr(f"{cfg.project_id}/HANDOFF/README_FOR_B.md", README_B)
        z.writestr(f"{cfg.project_id}/HANDOFF/README_FOR_C.md", README_C)
    return sorted(seen)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-checkpoint", action="store_true", help="leave the (large) .pt checkpoints out of the zip")
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args(argv)
    cfg = cfg_from_dir(a.project_dir)
    for E in edits(cfg):
        t = rebuild_comparison_md(E)
        if t:
            (E / "metrics").mkdir(exist_ok=True)
            (E / "metrics" / "rebuild_comparison.md").write_text(t + "\n")
    if a.tables_only:
        print(summary_md(cfg))
        return
    out = a.out or Path(f"handoff_{cfg.project_id}.zip")
    files = pack(cfg, out, include_ckpt=not a.no_checkpoint)
    print(f"[handoff] {out}  ({out.stat().st_size / 2**20:.1f} MB, {len(files)} files + HANDOFF/*.md)")


if __name__ == "__main__":
    main()
