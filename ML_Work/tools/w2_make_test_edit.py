#!/usr/bin/env python3
"""Make a FAKE edit (recolour the masked object) so the rebuild can be tested without Person B's diffusion stage.

    python tools/w2_make_test_edit.py --project-dir P --edit-id E [--color 0,0,255] [--dilate 6]

Writes edits/E/edited/<stem>.png for every present frame (pixels outside the dilated mask are bit-identical to the
frame, exactly B's guarantee) + edited/index.json {"edited": [{"name","file"}]}. Not part of the product: a test aid.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from common import masks as M  # noqa: E402
from common.w2_common import cfg_from_dir  # noqa: E402


def make(project: Path, edit_id: str, color=(0, 0, 255), dilate: int = 6, strength: float = 0.8) -> int:
    E = M.edit_dir(project, edit_id)
    idx = M.read_index(E / "masks" / "index.json")
    (E / "edited").mkdir(parents=True, exist_ok=True)
    items = []
    for f in idx["frames"]:
        if not f["present"]:
            continue
        img = cv2.imread(str(Path(project) / "frames" / "images" / f["name"]), cv2.IMREAD_COLOR)
        mk = M.load_mask(E / "masks" / (Path(f["name"]).stem + ".png"))
        out = img.copy()
        tint = (img * (1 - strength) + np.array(color, np.float32)[None, None] * strength).astype(np.uint8)
        out[mk] = tint[mk]
        _ = dilate          # the recolour stays inside the mask itself, which is inside the dilated one
        cv2.imwrite(str(E / "edited" / (Path(f["name"]).stem + ".png")), out)
        items.append({"name": f["name"], "file": f"edited/{Path(f['name']).stem}.png"})
    (E / "edited" / "index.json").write_text(json.dumps({"edited": items}, indent=2))
    return len(items)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", type=Path, required=True)
    ap.add_argument("--edit-id", required=True)
    ap.add_argument("--color", default="0,0,255", help="BGR, e.g. 0,0,255 = red")
    ap.add_argument("--dilate", type=int, default=6)
    a = ap.parse_args(argv)
    n = make(cfg_from_dir(a.project_dir).project, a.edit_id, tuple(int(x) for x in a.color.split(",")), a.dilate)
    print(f"[test-edit] wrote {n} edited frames to edits/{a.edit_id}/edited/")


if __name__ == "__main__":
    main()
