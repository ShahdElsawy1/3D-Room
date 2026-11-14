#!/usr/bin/env python3
"""Split a big Kaggle output (folder or .zip) into "commit this" and "host elsewhere".

GitHub rejects any file > 100 MB, warns above 50 MB, and the whole repo should stay well under 1 GB.
Kaggle outputs (handoff zip ~1 GB, e1_masks_edited.zip, demo_export.zip, checkpoints, .ply) do not fit.
This script never uploads anything. It

  1. copies the SMALL, useful files (json / md / csv / txt / png / jpg / .splat up to --max-mb) into --dest,
  2. writes <dest>/LARGE_FILES.md: every file it left out, with size + sha256 and an empty "link" column
     that you fill in after uploading it to Google Drive / Hugging Face / a GitHub Release,
  3. prints a size summary so you can see what would have broken the push.

Works on a zip WITHOUT extracting it fully (only the small files are read).

    python tools/pack_for_github.py --src handoff_demo.zip --dest results/week2/handoff
    python tools/pack_for_github.py --src demo_export.zip  --dest results/week1/export --max-mb 40
    python tools/pack_for_github.py --src ~/Downloads/e1_masks_edited.zip --dest results/week2/e1 --max-mb 5

Always left out: frames/ images/ depth/ ckpts/ gs/ poses/ scene/ folders, *.pt, *.ply, *.npy, *.npz, *.mp4, anything over --max-mb.
Use --keep-dir NAME to keep one of those folders.
"""
import argparse, hashlib, shutil, sys, zipfile
from pathlib import Path

KEEP_EXT = {".json", ".md", ".txt", ".csv", ".png", ".jpg", ".jpeg", ".splat", ".log"}
SKIP_DIRS = {"frames", "images", "depth", "conf", "ckpts", "gs", "poses", "scene", "sparse_depth", "week2"}  # bulk data: host elsewhere
NEVER = {".pt", ".pth", ".ply", ".safetensors", ".ckpt", ".npy", ".npz", ".mp4", ".bin"}


def sha256_stream(fh, chunk=1 << 20):
    h = hashlib.sha256()
    for b in iter(lambda: fh.read(chunk), b""):
        h.update(b)
    return h.hexdigest()


def entries(src: Path):
    """yield (relative name, size, opener) for a folder or a zip."""
    if src.is_file() and zipfile.is_zipfile(src):
        z = zipfile.ZipFile(src)
        for i in z.infolist():
            if not i.is_dir():
                yield i.filename, i.file_size, (lambda n=i.filename: z.open(n))
    else:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                yield str(p.relative_to(src)), p.stat().st_size, (lambda p=p: open(p, "rb"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder or .zip from Kaggle")
    ap.add_argument("--dest", required=True, help="folder inside the repo, e.g. results/week2/handoff")
    ap.add_argument("--max-mb", type=float, default=25.0, help="largest single file to keep (default 25; hard GitHub limit is 100)")
    ap.add_argument("--keep-dir", action="append", default=[], help="folder name to keep even though it is normally skipped (e.g. --keep-dir masks)")
    ap.add_argument("--max-total-mb", type=float, default=150.0, help="stop copying when the kept files reach this total")
    a = ap.parse_args()
    src, dest = Path(a.src).expanduser(), Path(a.dest)
    if not src.exists():
        sys.exit(f"not found: {src}")
    kept, left, total_all, total_kept = [], [], 0, 0
    for name, size, opener in entries(src):
        total_all += size
        ext = Path(name).suffix.lower()
        skipped = (set(Path(name).parts[:-1]) & SKIP_DIRS) - set(a.keep_dir)
        ok = not skipped and ext in KEEP_EXT and ext not in NEVER and size <= a.max_mb * 1e6 and total_kept + size <= a.max_total_mb * 1e6
        if ok:
            out = dest / name
            out.parent.mkdir(parents=True, exist_ok=True)
            with opener() as fi, open(out, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            kept.append((name, size)); total_kept += size
        else:
            left.append((name, size))
    # collapse huge folders (e.g. frames/) into one line each, so LARGE_FILES.md stays readable
    groups, singles = {}, []
    for name, size in left:
        parts = Path(name).parts
        if len(parts) > 2 and size < 50e6:
            groups.setdefault("/".join(parts[:2]) + "/", []).append(size)
        else:
            singles.append((name, size))
    dest.mkdir(parents=True, exist_ok=True)
    lines = ["# Files NOT stored in git", "",
             f"Source: `{src.name}`. Upload these to Drive / Hugging Face / a GitHub Release and paste the link.", "",
             "| file | size | sha256 | link |", "|---|---|---|---|"]
    for name, size in sorted(singles, key=lambda x: -x[1]):
        lines.append(f"| `{name}` | {size/1e6:,.1f} MB | - | _add link_ |")
    for g, sizes in sorted(groups.items()):
        lines.append(f"| `{g}` ({len(sizes)} files) | {sum(sizes)/1e6:,.1f} MB | - | _add link_ |")
    if src.is_file():  # hash of the archive itself is what people download
        with open(src, "rb") as fh:
            lines += ["", f"Archive `{src.name}` sha256: `{sha256_stream(fh)}` ({src.stat().st_size/1e6:,.1f} MB)"]
    (dest / "LARGE_FILES.md").write_text("\n".join(lines) + "\n")
    print(f"total in source : {total_all/1e6:10.1f} MB ({len(kept)+len(left)} files)")
    print(f"copied to {dest}: {total_kept/1e6:10.1f} MB ({len(kept)} files)")
    print(f"left out        : {(total_all-total_kept)/1e6:10.1f} MB ({len(left)} files) -> {dest/'LARGE_FILES.md'}")
    big = [n for n, s in kept if s > 50e6]
    if big:
        print("WARNING: files above 50 MB were kept:", big)


if __name__ == "__main__":
    main()
