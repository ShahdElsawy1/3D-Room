#!/usr/bin/env python3
"""Stand-in for Person A's tracking, ONLY for benchmarking the diffusion stage before real masks exist.

Writes edits/<eid>/masks/<stem>.png (a filled ellipse inside the given box) + index.json for the listed frames, in
the agreed format. Real edits use Person A's masks; the report must say when box masks were used.

  python make_box_masks.py --project-dir <root>/<pid> --edit-id bench --box 300,180,620,420 \
         --frames frame_000010.jpg,frame_000011.jpg,frame_000012.jpg
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=Path, required=True)
    ap.add_argument("--edit-id", required=True)
    ap.add_argument("--box", required=True, help="x0,y0,x1,y1 in frame pixels (same box for every frame)")
    ap.add_argument("--frames", required=True, help="comma-separated frame names, first = reference")
    a = ap.parse_args(argv)
    x0, y0, x1, y1 = map(int, a.box.split(","))
    names = a.frames.split(",")
    out = a.project_dir / "edits" / a.edit_id / "masks"
    out.mkdir(parents=True, exist_ok=True)
    all_names = sorted(p.name for p in (a.project_dir / "frames" / "images").iterdir() if p.suffix.lower() in (".jpg", ".png"))
    frames = []
    for nm in all_names:
        present = nm in names
        if present:
            w, h = Image.open(a.project_dir / "frames" / "images" / nm).size
            m = np.zeros((h, w), np.uint8)
            cv2.ellipse(m, ((x0 + x1) // 2, (y0 + y1) // 2), ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 255, -1)
            Image.fromarray(m).save(out / (Path(nm).stem + ".png"))
        frames.append({"name": nm, "present": present, "source": "box_stand_in",
                       "score": 1.0 if present else 0.0, "area_px": int((m > 0).sum()) if present else 0})
    (out / "index.json").write_text(json.dumps({"ref_frame": names[0], "method": "box_stand_in", "frames": frames}, indent=2))
    print(f"wrote {len(names)} masks to {out}")


if __name__ == "__main__":
    main()
