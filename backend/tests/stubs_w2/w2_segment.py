"""Stand-in for Person A's segmentation stage: an ellipse around the mean positive click.
Contract: --edit-id E (--frame F --clicks JSON | --add-frames JSON) --project-dir P [--backend-url --task-id]"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import ellipse_mask, mask_dir, parser, progress, read_index, write_index  # noqa: E402


def main():
    ap = parser()
    ap.add_argument("--frame")
    ap.add_argument("--clicks")
    ap.add_argument("--add-frames")
    a = ap.parse_args()
    if os.environ.get("STUB_W2_FAIL") == "segment":
        print("stub: SAM out of memory", file=sys.stderr)
        sys.exit(3)
    progress(a, "segmenting", 10)
    idx = read_index(a)
    jobs = json.loads(a.add_frames) if a.add_frames else [{"name": a.frame, "clicks": json.loads(a.clicks)}]
    for job in jobs:
        img = a.project_dir / "frames" / "images" / job["name"]
        pos = [c for c in job["clicks"] if c.get("positive", True)]
        w, h = __import__("PIL.Image", fromlist=["Image"]).open(img).size
        cx = sum(c["x"] for c in pos) // len(pos) if pos else w // 2
        cy = sum(c["y"] for c in pos) // len(pos) if pos else h // 2
        m = ellipse_mask(img, cx, cy)
        m.save(mask_dir(a) / (Path(job["name"]).stem + ".png"))
        idx["frames"] = [f for f in idx["frames"] if f["name"] != job["name"]]
        idx["frames"].append({"name": job["name"], "present": True, "source": "user" if a.add_frames else "click",
                              "score": 0.95, "area_px": int((__import__("numpy").asarray(m) > 0).sum())})
    if not a.add_frames:
        idx["ref_frame"] = a.frame
    write_index(a, idx)
    progress(a, "segmenting", 100)


if __name__ == "__main__":
    main()
