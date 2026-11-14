#!/usr/bin/env python3
"""Week 2 / A - find the clicked object in every other frame.

    python stages/w2_track.py --project-dir P --edit-id E [--tracker both|sam2|lift]

Inputs : frames/images, week2/inputs.json, scene/sparse/0 (poses), week2/depth (splat depth), edits/E/masks (the click
         mask = anchor; frames marked by the user, source "user", are anchors too).
Methods (PDF "simplest first", each evaluated by tools/w2_eval_tracking.py):
  1. sam2  : SAM 2.1 video propagation from the anchors, forwards and backwards.  Simple, great when views overlap
             a lot, can lose the object after a big viewpoint change / occlusion, or slide onto a look-alike.
  2. lift  : reference mask + splat depth -> 3D points -> project into every frame with a visibility check (an
             occluder hides the object, a frame that never sees it is "not present"), then SAM 2 refines the edges
             from box + points.  Geometry-driven: survives gaps, leave-and-return and viewpoint jumps.
  both     : per frame, take (1) when it agrees with the geometry (IoU >= --iou-agree with the lifted coarse mask) or
             when there is no geometry; otherwise take (2); if neither is confident the frame is "not present".
Output (contract with B/C): edits/E/masks/<stem>.png + index.json  (source: click | user | propagated | lifted).
User-marked frames are never overwritten. Per-frame decisions go to edits/E/metrics/track_debug.json.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from common import lift as L  # noqa: E402
from common import masks as M  # noqa: E402
from common import undist  # noqa: E402
from common.metrics import StageMetrics  # noqa: E402
from common.w2_common import (base_parser, cfg_from_dir, dump_json, edit_progress, load_cams, load_depth,  # noqa: E402
                              load_inputs, read_rgb, space_maps)

STAGE = "tracking"


def run_tracking(cfg, edit_id, inputs, *, tracker="both", video_tracker=None, refiner=None, out_sub="masks",
                 min_conf=0.6, min_visible=0.15, iou_agree=0.3, progress=None) -> dict:
    """The whole tracking logic with the models injected, so it runs (and is tested) without a GPU.
       video_tracker(anchors{idx: mask}, start_idx, progress) -> {idx: (mask, conf)}       [frames space]
       refiner(rgb, prompts{box,pos,neg}) -> (mask, score)                                   [frames space]"""
    E = M.edit_dir(cfg.project, edit_id)
    names = M.frame_names(cfg.project)
    src_idx = M.read_index(E / "masks" / "index.json")
    ref = src_idx.get("ref_frame")
    if not ref or ref not in names:
        raise SystemExit("no reference mask yet - run w2_segment first")
    anchors = {}
    for f in src_idx["frames"]:
        if f["present"] and f["source"] in ("click", "user"):
            m = M.load_mask(E / "masks" / (Path(f["name"]).stem + ".png"))
            if m is not None:
                anchors[names.index(f["name"])] = m
    ref_i = names.index(ref)
    f2s, s2f = space_maps(cfg, inputs)

    # ---- method 1: SAM 2 video propagation --------------------------------------------------------------------
    vid, t_vid = {}, 0.0
    if tracker in ("both", "sam2"):
        if video_tracker is None:
            raise RuntimeError("sam2 tracker requested but no video model was given")
        t0 = time.perf_counter()
        vid = video_tracker(anchors, ref_i, (lambda f: progress(0.45 * f)) if progress else None)
        t_vid = time.perf_counter() - t0

    # ---- method 2: lift to 3D + project + (refine) --------------------------------------------------------------
    geo, t_geo, geo_note = {}, 0.0, None
    if tracker in ("both", "lift"):
        t0 = time.perf_counter()
        cams = load_cams(cfg)
        depth_ref = load_depth(cfg, inputs, ref)
        if ref not in cams or depth_ref is None:
            geo_note = ("no splat depth / pose for the reference frame - lifting skipped (render depth: "
                        "stages/s6_prepare_week2.py --render-depth on, needs the checkpoint)")
            if tracker == "lift":
                raise SystemExit(geo_note)
            print(f"[w2_track] {geo_note}", flush=True)
        else:
            m_ref = anchors[ref_i]
            if f2s is not None:                               # frames space -> scene space (undistorted)
                m_ref = undist.remap(m_ref.astype(np.uint8) * 255, f2s, nearest=True) > 0
            pts = L.unproject_mask(m_ref, depth_ref, cams[ref])
            geo_note = f"{len(pts)} 3D points from the reference mask"
            for k, n in enumerate(names):
                if n not in cams:
                    continue
                if names.index(n) in anchors:
                    continue
                out = L.lift_to_frame(pts, cams[n], load_depth(cfg, inputs, n), (cams[n].height, cams[n].width),
                                      close_frac=0.012)
                coarse = out["mask"]
                if s2f is not None:
                    coarse = undist.remap(coarse.astype(np.uint8) * 255, s2f, nearest=True) > 0
                out["coarse"] = coarse
                geo[names.index(n)] = out
                if progress:
                    progress(0.45 + 0.25 * (k + 1) / len(names))
        t_geo = time.perf_counter() - t0
        # Sanity check: geometry that contradicts SAM 2 on (nearly) every frame SAM 2 is confident about is broken
        # (typically the splat depth was rendered in another world space - run stages/w2_calibrate.py), not "the
        # object is absent". Without this a broken depth vetoes every good SAM 2 mask (Week 2 run: 1/61 frames).
        if geo and vid:
            both_ = [i for i in geo if i in vid and vid[i][0].any() and vid[i][1] >= min_conf]
            if len(both_) >= 3 and sum(not geo[i]["coarse"].any() for i in both_) / len(both_) > 0.9:
                geo_note = (f"{geo_note}; IGNORED: it contradicts SAM 2 on {len(both_)} confident frames "
                            "(poses/depth space mismatch? run stages/w2_calibrate.py)")
                print(f"[w2_track] WARNING geometry {geo_note}", flush=True)
                geo = {}

    # ---- refine the lifted masks with SAM 2 (box + points from the coarse mask) ----------------------------------
    t0 = time.perf_counter()
    for k, (i, g) in enumerate(sorted(geo.items())):
        g["refined"], g["score"] = None, 0.0
        if g["visible_frac"] >= min_visible and g["coarse"].any():
            if refiner is not None:
                pr = L.prompts_from_mask(g["coarse"])
                if pr is not None:
                    mk, sc = refiner(read_rgb(cfg.images_dir / names[i]), pr)
                    g["refined"], g["score"] = M.clean_mask(mk), float(sc)
            else:
                g["refined"], g["score"] = g["coarse"], 0.5 * g["visible_frac"]
        if progress:
            progress(0.70 + 0.25 * (k + 1) / max(len(geo), 1))
    t_ref = time.perf_counter() - t0

    # ---- decide per frame ------------------------------------------------------------------------------------------
    entries, debug, saved = [], {}, {}
    for i, n in enumerate(names):
        stem = Path(n).stem
        if i in anchors:
            f = next(x for x in src_idx["frames"] if x["name"] == n)
            entries.append(f)
            debug[n] = {"decision": "anchor", "source": f["source"]}
            continue
        a = vid.get(i)
        a_ok = a is not None and a[0].any() and a[1] >= min_conf
        g = geo.get(i)
        g_vis = g is not None and g["visible_frac"] >= min_visible and g.get("refined") is not None and g["refined"].any()
        d = {"sam2_conf": None if a is None else round(float(a[1]), 3),
             "visible_frac": None if g is None else round(g["visible_frac"], 3),
             "in_view_frac": None if g is None else round(g["in_view_frac"], 3)}
        mask, source, score = None, None, 0.0
        if tracker == "sam2":
            if a_ok:
                mask, source, score = a[0], "propagated", a[1]
        elif tracker == "lift":
            if g_vis:
                mask, source, score = g["refined"], "lifted", g["score"]
        else:
            if a_ok and g is not None and g["coarse"].any():
                agree = M.iou(a[0], g["coarse"])
                d["iou_with_geometry"] = round(agree, 3)
                if agree >= iou_agree:
                    mask, source, score = a[0], "propagated", a[1]
                elif g_vis:
                    mask, source, score = g["refined"], "lifted", g["score"]
                    d["note"] = "sam2 disagrees with the geometry -> lifted"
                else:
                    d["note"] = "sam2 disagrees and geometry is not confident -> not present"
            elif a_ok and g is None:
                mask, source, score = a[0], "propagated", a[1]          # no geometry for this frame: trust SAM 2
            elif a_ok and g is not None:
                d["note"] = "geometry says the object is not visible -> not present"
            elif g_vis:
                mask, source, score = g["refined"], "lifted", g["score"]
        d["decision"] = source or "absent"
        debug[n] = d
        if mask is not None and mask.any():
            mask = M.clean_mask(mask)
            area = M.save_mask(E / out_sub / (stem + ".png"), mask)
            entries.append(M.entry(n, area > 0, source, score, area))
        else:
            p = E / out_sub / (stem + ".png")
            if p.exists():
                p.unlink()
            entries.append(M.entry(n, False, None, 0.0, 0))
    method = {"both": "fused_sam2_lift", "sam2": "sam2_video", "lift": "lift3d_sam2"}[tracker]
    if out_sub != "masks":                                    # evaluation variants keep their own copy of the anchors
        for i, m in anchors.items():
            M.save_mask(E / out_sub / (Path(names[i]).stem + ".png"), m)
    idx = M.write_index(E / out_sub / "index.json", {"ref_frame": ref, "method": method, "frames": entries}, names)
    stats = {"tracker": tracker, "present": sum(f["present"] for f in idx["frames"]), "frames": len(names),
             "by_source": {s: sum(1 for f in idx["frames"] if f["present"] and f["source"] == s)
                           for s in M.SOURCES if any(f["present"] and f["source"] == s for f in idx["frames"])},
             "seconds": {"sam2_video": round(t_vid, 2), "lift": round(t_geo, 2), "refine": round(t_ref, 2)},
             "geometry": geo_note, "thresholds": {"min_conf": min_conf, "min_visible": min_visible,
                                                  "iou_agree": iou_agree}}
    dump_json(E / "metrics" / ("track_debug.json" if out_sub == "masks" else f"track_debug_{Path(out_sub).parent.name}.json"),
              {"summary": stats, "frames": debug})
    return stats


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--tracker", default="both", choices=["both", "sam2", "lift"])
    ap.add_argument("--out", default="masks", help="sub-folder of edits/E for the result (evaluation uses others)")
    ap.add_argument("--min-conf", type=float, default=0.6, help="SAM 2 video confidence needed to accept a mask")
    ap.add_argument("--min-visible", type=float, default=0.15, help="share of the 3D object that must be visible")
    ap.add_argument("--iou-agree", type=float, default=0.3, help="SAM 2 mask vs lifted coarse mask: IoU to agree")
    ap.add_argument("--no-refine", action="store_true", help="use the coarse lifted mask (no SAM 2 edge refinement)")
    args = ap.parse_args(argv)
    cfg = cfg_from_dir(args.project_dir)
    prog = edit_progress(cfg, args.edit_id, args)
    prog.report(STAGE, 0.0, force=True)
    inputs = load_inputs(cfg)
    if inputs.get("_inferred"):
        print("[w2_track] week2/inputs.json missing - run stages/s6_prepare_week2.py (lifting needs the depth)", flush=True)
    E = M.edit_dir(cfg.project, args.edit_id)
    with StageMetrics(E / "metrics", "w2_track") as m:
        try:
            from common.sam_wrap import SamImage, SamVideo
            video = SamVideo(args.sam_model, args.device, args.sam_dtype) if args.tracker in ("both", "sam2") else None
            img = SamImage(args.sam_model, args.device, args.sam_dtype) if (
                args.tracker in ("both", "lift") and not args.no_refine) else None

            def video_tracker(anchors, start, cb):
                return video.propagate(cfg.images_dir, anchors, start, cb)

            def refiner(rgb, pr):
                pts = [(x, y, 1) for x, y in pr["pos"]] + [(x, y, 0) for x, y in pr["neg"]]
                return img.segment(rgb, pts, box=pr["box"], multimask=False)

            stats = run_tracking(cfg, args.edit_id, inputs, tracker=args.tracker,
                                 video_tracker=video_tracker if video else None, refiner=refiner if img else None,
                                 out_sub=args.out, min_conf=args.min_conf, min_visible=args.min_visible,
                                 iou_agree=args.iou_agree, progress=lambda f: prog.report(STAGE, f))
            m.extra.update(stats)
            probs = M.validate(cfg.project, args.edit_id, args.out)
            if probs:
                print("[w2_track] CONTRACT PROBLEMS:", probs[:5], flush=True)
            print(f"[w2_track] {stats['present']}/{stats['frames']} frames present, by source {stats['by_source']}, "
                  f"seconds {stats['seconds']}", flush=True)
            prog.report(STAGE, 1.0, force=True)
        except Exception as e:
            prog.fail(STAGE, e)
            raise


if __name__ == "__main__":
    main()
