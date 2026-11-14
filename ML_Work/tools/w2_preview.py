#!/usr/bin/env python3
"""Contact sheet of the masks (overlay on the frames) so you can see what segmentation/tracking produced.

    python tools/w2_preview.py --project-dir P --edit-id E [--sub masks] [--cols 6] [--out edits/E/preview.png]
Green = mask from a click/user, blue = propagated, orange = lifted; red border = object not present.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from common import masks as M  # noqa: E402
from common.w2_common import cfg_from_dir  # noqa: E402

COLORS = {"click": (0, 220, 0), "user": (0, 220, 0), "propagated": (255, 120, 0), "lifted": (0, 150, 255),
          "reid": (200, 0, 200)}          # BGR


def overlay(img, mask, color, alpha=0.5):
    out = img.copy()
    if mask is not None and mask.any():
        out[mask] = (img[mask] * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)
        cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnt, -1, color, 2)
    return out


def sheet(project: Path, edit_id: str, sub: str = "masks", cols: int = 6, thumb_w: int = 240) -> np.ndarray:
    E = M.edit_dir(project, edit_id)
    idx = M.read_index(E / sub / "index.json")
    tiles = []
    for f in idx["frames"]:
        img = cv2.imread(str(Path(project) / "frames" / "images" / f["name"]), cv2.IMREAD_COLOR)
        mk = M.load_mask(E / sub / (Path(f["name"]).stem + ".png")) if f["present"] else None
        t = overlay(img, mk, COLORS.get(f["source"], (255, 255, 255)))
        h = int(t.shape[0] * thumb_w / t.shape[1])
        t = cv2.resize(t, (thumb_w, h), interpolation=cv2.INTER_AREA)
        if not f["present"]:
            cv2.rectangle(t, (0, 0), (thumb_w - 1, h - 1), (0, 0, 255), 4)
        cv2.putText(t, f"{Path(f['name']).stem} {f['source'] or '-'}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    if not tiles:
        raise SystemExit("no frames")
    h, w = tiles[0].shape[:2]
    rows = -(-len(tiles) // cols)
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        canvas[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = t
    return canvas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", type=Path, required=True)
    ap.add_argument("--edit-id", required=True)
    ap.add_argument("--sub", default="masks")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    cfg = cfg_from_dir(a.project_dir)
    out = a.out or M.edit_dir(cfg.project, a.edit_id) / f"preview_{Path(a.sub).parent.name or a.sub}.png"
    cv2.imwrite(str(out), sheet(cfg.project, a.edit_id, a.sub, a.cols))
    print(f"[preview] {out}")


if __name__ == "__main__":
    main()
