#!/usr/bin/env python3
"""Generate the Week 1 deliverables that come from measurements (no hand-copying numbers):

    python tools/make_reports.py --project demo

  docs/generated/VRAM_RUNTIME_TABLE.md   "a VRAM and runtime table for every stage"
  docs/generated/WEEK2_HANDOFF.md        "one-page Week 2 handoff: where frames, poses, intrinsics and the splat are"
  <project>/manifest.json                machine-readable version of the handoff
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.colmap_io import read_model  # noqa: E402
from common.config import add_common_args, config_from_args, read_json, write_json  # noqa: E402

STAGES = [("s1_extract_frames", "1. Frame extraction (ffmpeg + blur filter)"),
          ("s2a_poses_vggt", "2a. Poses: VGGT (chunked, fp16)"),
          ("s2b_poses_glomap", "2b. Poses: GLOMAP (features + seq. matching + mapper)"),
          ("s4_train_3dgs", "4. 3DGS training"),
          ("s5_export_splat", "5. Export (.ply -> .splat)")]


def fmt(v, nd=1):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--out", default="docs/generated")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ VRAM / runtime table
    rows = ["| stage | wall time (s) | peak VRAM, nvidia-smi (MiB) | peak VRAM, torch alloc (MiB) | notes |",
            "|---|---|---|---|---|"]
    for key, label in STAGES:
        m = read_json(cfg.metrics_dir / f"{key}.json")
        if not m:
            rows.append(f"| {label} | not run | | | |")
            continue
        notes = []
        for k in ("frames_kept", "registered_frames", "num_gaussians", "iters", "chunks", "dtype", "splat_mb"):
            if m.get(k) is not None:
                notes.append(f"{k}={m[k]}")
        rows.append(f"| {label} | {fmt(m.get('seconds'))} | {fmt(m.get('peak_vram_mb'), 0)} | "
                    f"{fmt(m.get('torch_peak_allocated_mb'), 0)} | {', '.join(notes)} |")
    sweep = read_json(cfg.metrics_dir / "vggt_memory_sweep.json")
    table = ["# VRAM and runtime per stage", "", *rows]
    if sweep:
        table += ["", "## VGGT peak VRAM vs frames per chunk", "", f"dtype `{sweep['dtype']}`, input {sweep['size_wh']}", "",
                  "| frames | seconds | peak alloc (MiB) |", "|---|---|---|"]
        for r in sweep["rows"]:
            table.append(f"| {r['frames']} | {r.get('seconds', 'OOM')} | {r.get('peak_alloc_mb', r.get('error', ''))} |")
    train = read_json(cfg.gs_dir / "latest.json")
    if train:
        table += ["", "## 3DGS run", "", f"- trainer: {train.get('trainer')}, iterations: {train.get('iters')}",
                  f"- training wall time: {train.get('train_wall_seconds')} s",
                  f"- final #Gaussians: {train.get('num_gaussians')}",
                  f"- trainer-reported peak memory: {train.get('trainer_peak_mem_gb')} GB",
                  f"- held-out metrics: {train.get('heldout')}"]
    (out / "VRAM_RUNTIME_TABLE.md").write_text("\n".join(table) + "\n")

    # ------------------------------------------------ handoff
    chosen = read_json(cfg.scene_dir / "chosen.json", {})
    idx = read_json(cfg.frames_index, {})
    R = cfg.rel                                    # every path below is relative to the project folder
    manifest = {
        "project": cfg.project_id, "root": ".", "paths_are_relative_to": "the project folder",
        "pose_method": chosen.get("method"),
        "frames": {"dir": R(cfg.images_dir), "count": idx.get("kept"), "size_wh": idx.get("image_size_wh"),
                   "index": R(cfg.frames_index), "sampled_fps": idx.get("fps"),
                   "naming": "000001.jpg ... (1-based, zero-padded; identical names in the COLMAP model)"},
        "colmap": {"scene_dir": R(cfg.scene_dir), "sparse": R(cfg.scene_dir / "sparse" / "0"),
                   "formats": ["bin", "txt"], "convention": "world->camera (OpenCV: +x right, +y down, +z forward); "
                   "qvec = (w,x,y,z)"},
        "depth": ({"dir": R(cfg.poses_dir("vggt") / "depth"), "conf_dir": R(cfg.poses_dir("vggt") / "conf"),
                   "format": ".npy float16 z-depth at VGGT resolution, same world scale as sparse/0 (only valid when "
                             "pose_method == vggt; otherwise use week2/depth rendered from the splat)",
                   "meta": R(cfg.poses_dir("vggt") / "vggt_meta.json")}
                  if (cfg.poses_dir("vggt") / "depth").exists() else None),
        "splat": {"ply": R(cfg.export_dir / "scene.ply"), "splat": R(cfg.export_dir / "scene.splat"),
                  "cameras": R(cfg.export_dir / "cameras.json"),
                  "meta": R(cfg.export_dir / "scene_meta.json"),
                  "trainer_output": R(cfg.gs_dir)},
        "week2_inputs": R(cfg.p("week2", "inputs.json")) if cfg.p("week2", "inputs.json").exists() else None,
        "metrics_dir": R(cfg.metrics_dir),
    }
    intr = None
    if (cfg.scene_dir / "sparse" / "0").exists():
        model = read_model(cfg.scene_dir / "sparse" / "0")
        manifest["intrinsics"] = {str(cid): {"model": c.model, "width": c.width, "height": c.height, "params": c.params.tolist()}
                                  for cid, c in model.cameras.items()}
        manifest["registered_frames"] = len(model.images)
        intr = manifest["intrinsics"]
    write_json(cfg.manifest_path, manifest)

    ho = [
        "# Week 2 handoff (Person A)", "",
        "Everything below is also in `manifest.json`. All paths are relative to the project folder "
        f"`{cfg.project}`.", "",
        "| what | where | format |", "|---|---|---|",
        "| original video | `video/input.*` | as uploaded |",
        f"| frames ({idx.get('kept')} @ {idx.get('image_size_wh')}) | `frames/images/000001.jpg ...` | JPEG, resized, blur-filtered |",
        "| frame metadata | `frames/frames_index.json` | name, source video time, blur score |",
        f"| camera poses ({chosen.get('method')}) | `scene/sparse/0/images.bin` (+ `.txt`) | COLMAP, world->camera, OpenCV axes |",
        "| intrinsics | `scene/sparse/0/cameras.bin` (+ `.txt`) | COLMAP PINHOLE (fx fy cx cy), at frame resolution |",
        "| sparse points | `scene/sparse/0/points3D.bin` (+ `.txt`), `scene/points.ply` | COLMAP / PLY |",
        "| per-frame depth + confidence | `poses/vggt/depth/*.npy`, `poses/vggt/conf/*.npy` | float16, VGGT resolution |",
        "| trained splat | `export/scene.ply`, `export/scene.splat` | 3DGS PLY / compact .splat |",
        "| training cameras for the viewer | `export/cameras.json` | position + c2w rotation + fx fy cx cy |",
        "| trainer checkpoint | `gs/<trainer>_<iters>/ckpts/*.pt` (gsplat) | for fine-tuning in Week 2 (4.4) |",
        "| **Week 2 inputs (start here)** | `week2/inputs.json` | image spaces + sizes, held-out split, depth sources, "
        "checkpoint, `required_files` to upload |",
        "| sparse depth per frame | `week2/sparse_depth/*.npy` | (N,3) [u, v, z] of visible COLMAP points, scene/images pixels |",
        "| splat-rendered depth (GPU run) | `week2/depth/*.npy` | float16, valid for any pose method |",
        "| measurements | `metrics/*.json` | per-stage time + VRAM |", "",
        "## Things Week 2 must know", "",
        "- Frame `N` in `frames/images` has the SAME name in the COLMAP model. Use it as the key everywhere "
        "(masks, edited frames).",
        "- **Image space:** if the GLOMAP scene was undistorted, `scene/images` is a slightly different size/pixels than "
        "`frames/images` (see `week2/inputs.json -> images`). Masks and edits must be made on `scene/images` "
        "(`recommended_edit_space`) or they will not line up with the splat's cameras.",
        "- All paths in the JSON files are relative to the project folder; never use absolute paths from them.",
        "- Do not re-run pose estimation (Week 2 4.4): reuse `scene/sparse/0` and swap images.",
        "- **Lift to 3D / reprojection (4.2 method 3):** back-project masks with `poses/vggt/depth` + the VGGT-resolution "
        "intrinsics in `poses/vggt/vggt_meta.json` (`intrinsics_vggt_res`) and the poses in `scene/sparse/0` "
        "(`common/geometry.py::unproject_depth`, `project`). Valid when the scene's pose method is VGGT.",
        "- Frames are 2-multiples, not 8/16-multiples: Week 2 diffusion must resize/pad and map back.",
        "- World frame has +y down (COLMAP). The viewer needs up = (0,-1,0) or a y-flip.",
        "- `export/cameras.json` lets the viewer snap to a training camera for the 'click on a reference frame' baseline.",
        "- gsplat holds out every 8th frame (`test_every`) for PSNR/SSIM; those frames are not in the splat's training set.",
    ]
    if intr:
        ho += ["", "## Intrinsics", "", "```json", __import__("json").dumps(intr, indent=2), "```"]
    (out / "WEEK2_HANDOFF.md").write_text("\n".join(ho) + "\n")
    print(f"wrote {out}/VRAM_RUNTIME_TABLE.md, {out}/WEEK2_HANDOFF.md, {cfg.manifest_path}")


if __name__ == "__main__":
    main()
