"""Stand-in for Person A's tracking stage: the reference mask copied to every frame except the last one
(simulates the object leaving the view). Contract: --edit-id E --project-dir P [--backend-url --task-id]"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import frames, mask_dir, parser, progress, read_index, write_index  # noqa: E402


def main():
    a = parser().parse_args()
    idx = read_index(a)
    ref = idx["ref_frame"]
    src = mask_dir(a) / (Path(ref).stem + ".png")
    names = frames(a)
    out = []
    for i, n in enumerate(names):
        progress(a, "tracking", 100 * i / len(names))
        present = i < len(names) - 1 or n == ref
        if present and n != ref:
            shutil.copyfile(src, mask_dir(a) / (Path(n).stem + ".png"))
        out.append({"name": n, "present": present, "source": "click" if n == ref else "propagated",
                    "score": 0.9 if present else 0.1, "area_px": 9000 if present else 0})
    idx.update(method="stub_copy", frames=out)
    write_index(a, idx)


if __name__ == "__main__":
    main()
