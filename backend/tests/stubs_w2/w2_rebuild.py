"""Stand-in for the rebuild stage (Person C with A): checks its inputs and writes a 'rebuilt' scene under
edits/E/ with Week 1's file names. Contract: --edit-id E --mode finetune|scratch --project-dir P [...]"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import parser, progress  # noqa: E402


def main():
    ap = parser()
    ap.add_argument("--mode", choices=["finetune", "scratch"], required=True)
    a = ap.parse_args()
    P, E = a.project_dir, a.project_dir / "edits" / a.edit_id
    edited = json.loads((E / "edited" / "index.json").read_text())["edited"]
    assert (P / "scene" / "sparse" / "0").is_dir(), "poses are reused: scene/sparse/0 must be downloaded"
    assert all((E / e["file"]).is_file() for e in edited)
    t = time.time()
    progress(a, "training", 50)
    (E / "export").mkdir(parents=True, exist_ok=True)
    (E / "export" / "scene.splat").write_bytes(b"\2" * 32 * 7)
    cams = json.loads((P / "export" / "cameras.json").read_text()) if (P / "export" / "cameras.json").exists() else [
        {"id": 0, "img_name": "x", "width": 1, "height": 1, "position": [0, 0, 0],
         "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "fx": 1, "fy": 1, "cx": 0, "cy": 0}]
    (E / "export" / "cameras.json").write_text(json.dumps(cams))
    (E / "export" / "scene_meta.json").write_text(json.dumps({"num_gaussians": 7, "trainer": "stub",
                                                               "iters": 3000 if a.mode == "finetune" else 7000}))
    (E / "metrics").mkdir(exist_ok=True)
    (E / "metrics" / "s4_rebuild.json").write_text(json.dumps({"seconds": round(time.time() - t, 3),
                                                               "peak_vram_mb": 0, "mode": a.mode,
                                                               "edited_frames": len(edited)}))
    progress(a, "exporting", 100)


if __name__ == "__main__":
    main()
