#!/usr/bin/env python3
"""Evaluate the tracking methods against a few hand-labelled frames (the Week 2 "tracking report").

    python tools/w2_eval_tracking.py --project-dir P --edit-id E --gt gt.json [--variants sam2 lift both]

gt.json - label 10-30 frames spread over the video (include frames where the object is NOT visible):
    {"000010.jpg": {"clicks": [{"x": 412, "y": 233, "positive": true}, ...]},      # SAM builds the GT mask from clicks
     "000055.jpg": {"absent": true}}                                               # object not in this frame
or put your own masks in edits/E/gt/<stem>.png and list them as {"000010.jpg": {"mask": true}}.

Variants are the folders written by `w2_track.py --out track_eval/<variant>/masks` ("fused" = edits/E/masks).
Output: edits/E/metrics/tracking_report.{json,md}: mean IoU on frames where the object is present, presence accuracy
(is-it-there decisions), false-present / missed counts, time per method.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from common import masks as M  # noqa: E402
from common.w2_common import cfg_from_dir, clicks_to_points, dump_json, read_rgb  # noqa: E402


def variant_dir(E: Path, v: str) -> Path:
    return E / "masks" if v in ("both", "fused") else E / "track_eval" / v / "masks"


def evaluate(E: Path, variants: list[str], gt: dict[str, np.ndarray | None]) -> dict:
    """gt: {frame_name: bool mask (object present) | None (object absent)} -> metrics per variant."""
    out = {}
    for v in variants:
        d = variant_dir(E, v)
        idx = M.read_index(d / "index.json")
        pres = {f["name"]: f["present"] for f in idx["frames"]}
        ious, tp = [], dict(present_ok=0, present_missed=0, absent_ok=0, absent_false=0)
        per = {}
        for n, g in gt.items():
            p = pres.get(n, False)
            pm = M.load_mask(d / (Path(n).stem + ".png")) if p else None
            if g is None:
                tp["absent_ok" if not p else "absent_false"] += 1
                per[n] = {"gt": "absent", "pred": "present" if p else "absent"}
            elif not p:
                tp["present_missed"] += 1
                ious.append(0.0)
                per[n] = {"gt": "present", "pred": "absent", "iou": 0.0}
            else:
                tp["present_ok"] += 1
                ious.append(M.iou(pm, g))
                per[n] = {"gt": "present", "pred": "present", "iou": round(ious[-1], 3)}
        n_gt = max(len(gt), 1)
        out[v] = {"mean_iou_present": round(float(np.mean(ious)), 4) if ious else None,
                  "median_iou_present": round(float(np.median(ious)), 4) if ious else None,
                  "presence_accuracy": round((tp["present_ok"] + tp["absent_ok"]) / n_gt, 4), **tp,
                  "method": idx.get("method"), "per_frame": per}
    return out


def to_markdown(res: dict, meta: dict) -> str:
    rows = ["| method | mean IoU (object present) | median IoU | presence accuracy | missed | false present |",
            "|---|---|---|---|---|---|"]
    for v, r in res.items():
        rows.append(f"| {v} | {r['mean_iou_present']} | {r['median_iou_present']} | {r['presence_accuracy']} | "
                    f"{r['present_missed']} | {r['absent_false']} |")
    return (f"# Tracking evaluation - edit `{meta['edit_id']}`\n\nLabelled frames: {meta['n_gt']} "
            f"({meta['n_absent']} with the object absent).\n\n" + "\n".join(rows) +
            "\n\nIoU = intersection over union with the hand-labelled mask; presence accuracy = correct "
            "'is the object in this frame' decisions. `both` is the fused result delivered in `masks/`.\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", type=Path, required=True)
    ap.add_argument("--edit-id", required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=["sam2", "lift", "both"])
    ap.add_argument("--sam-model", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--sam-dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    cfg = cfg_from_dir(a.project_dir)
    E = M.edit_dir(cfg.project, a.edit_id)
    spec = json.loads(a.gt.read_text())
    gt, sam = {}, None
    for n, s in spec.items():
        if s.get("absent"):
            gt[n] = None
        elif s.get("mask"):
            gt[n] = M.load_mask(E / "gt" / (Path(n).stem + ".png"))
        else:
            if sam is None:
                from common.sam_wrap import SamImage
                sam = SamImage(a.sam_model, a.device, a.sam_dtype)
            m, _ = sam.segment(read_rgb(cfg.images_dir / n), clicks_to_points(s["clicks"]))
            gt[n] = M.clean_mask(m)
            M.save_mask(E / "gt" / (Path(n).stem + ".png"), gt[n])
    res = evaluate(E, a.variants, gt)
    meta = {"edit_id": a.edit_id, "n_gt": len(gt), "n_absent": sum(g is None for g in gt.values())}
    dump_json(E / "metrics" / "tracking_report.json", {"meta": meta, "results": res})
    (E / "metrics").mkdir(exist_ok=True)
    (E / "metrics" / "tracking_report.md").write_text(to_markdown(res, meta))
    print((E / "metrics" / "tracking_report.md").read_text())


if __name__ == "__main__":
    main()
