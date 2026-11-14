#!/usr/bin/env python3
"""Week 2 / A - click(s) -> object mask on one frame (SAM 2.1), plus the fallback "mark the frames tracking missed".

    python stages/w2_segment.py --project-dir P --edit-id E --frame 000012.jpg \\
        --clicks '[{"x":412,"y":233,"positive":true},{"x":90,"y":40,"positive":false}]'
    python stages/w2_segment.py --project-dir P --edit-id E --add-frames '[{"name":"000030.jpg","clicks":[...]}]'

Coordinates are pixels of frames/images/<frame> (what the user sees). Output (contract with B/C):
    edits/E/masks/<stem>.png     8-bit 0/255, exactly the frame size
    edits/E/masks/index.json     one entry per frame; the other frames' entries/masks are kept
`--frame` sets the reference frame (source "click"); `--add-frames` marks extra frames (source "user").
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json  # noqa: E402

from common import masks as M  # noqa: E402
from common.metrics import StageMetrics  # noqa: E402
from common.w2_common import base_parser, cfg_from_dir, clicks_to_points, edit_progress, read_rgb  # noqa: E402

STAGE = "segmenting"


def segment_frames(cfg, edit_id: str, jobs: list[dict], source: str, segmenter, set_ref: str | None = None) -> dict:
    """jobs: [{"name": F, "clicks": [...]}]. `segmenter(rgb, points) -> (mask, score)` (SAM 2 or a fake in tests)."""
    md = M.edit_dir(cfg.project, edit_id) / "masks"
    names = M.frame_names(cfg.project)
    idx = M.read_index(md / "index.json")
    for job in jobs:
        name = job["name"]
        if name not in names:
            raise ValueError(f"unknown frame {name!r}")
        pts = clicks_to_points(job["clicks"])
        if not any(p[2] == 1 for p in pts):
            raise ValueError(f"{name}: need at least one positive click")
        rgb = read_rgb(cfg.images_dir / name)
        mask, score = segmenter(rgb, pts)
        mask = M.clean_mask(mask)
        if mask.shape != rgb.shape[:2]:
            raise RuntimeError(f"mask {mask.shape} != frame {rgb.shape[:2]}")
        area = M.save_mask(md / (Path(name).stem + ".png"), mask)
        idx["frames"] = [f for f in idx["frames"] if f["name"] != name]
        idx["frames"].append(M.entry(name, area > 0, source, score, area))
    if set_ref:
        idx["ref_frame"] = set_ref
    idx["method"] = idx.get("method") or "click"
    return M.write_index(md / "index.json", idx, names)


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--frame")
    ap.add_argument("--clicks")
    ap.add_argument("--add-frames")
    args = ap.parse_args(argv)
    if bool(args.add_frames) == bool(args.frame and args.clicks):
        ap.error("give either --frame F --clicks JSON, or --add-frames JSON")
    cfg = cfg_from_dir(args.project_dir)
    prog = edit_progress(cfg, args.edit_id, args)
    prog.report(STAGE, 0.05, force=True)
    jobs = json.loads(args.add_frames) if args.add_frames else [{"name": args.frame, "clicks": json.loads(args.clicks)}]
    with StageMetrics(M.edit_dir(cfg.project, args.edit_id) / "metrics", "w2_segment") as m:
        try:
            from common.sam_wrap import SamImage
            t0 = time.perf_counter()
            sam = SamImage(args.sam_model, args.device, args.sam_dtype)
            load_s = time.perf_counter() - t0
            prog.report(STAGE, 0.4)
            t0 = time.perf_counter()
            idx = segment_frames(cfg, args.edit_id, jobs, "user" if args.add_frames else "click",
                                 lambda rgb, pts: sam.segment(rgb, pts), None if args.add_frames else args.frame)
            m.extra.update(model=args.sam_model, dtype=args.sam_dtype, frames=[j["name"] for j in jobs],
                           model_load_seconds=round(load_s, 2), predict_seconds=round(time.perf_counter() - t0, 2))
            done = [f for f in idx["frames"] if f["name"] in {j["name"] for j in jobs}]
            print(f"[w2_segment] {[(f['name'], f['area_px'], f['score']) for f in done]}", flush=True)
            prog.report(STAGE, 1.0, force=True)
        except Exception as e:
            prog.fail(STAGE, e)
            raise


if __name__ == "__main__":
    main()
