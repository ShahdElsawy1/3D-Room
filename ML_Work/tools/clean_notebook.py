#!/usr/bin/env python3
"""Make an executed Kaggle notebook small enough for GitHub, keeping the useful outputs.

Why: a Kaggle run prints thousands of progress-bar / download lines and embeds multi-MB plots, so a
finished notebook easily reaches 8 MB and GitHub then fails to render it. This script
  * removes ANSI colour codes, tqdm / download bars and "[progress] ..." lines,
  * collapses repeated lines and keeps only the head and tail of very long outputs,
  * drops embedded images larger than --max-image-kb (save those as separate files instead),
  * removes metadata.widgets (the usual reason GitHub shows "Invalid Notebook").
Use --strip-all to delete every output (same effect as nbstripout).

    python tools/clean_notebook.py week2-done.ipynb notebooks/executed/week2_executed.ipynb
"""
import argparse, json, re, sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
NOISE = re.compile(r"\[progress\]|it/s\]|s/it\]|[0-9.]+ ?[kMG]?B/s|downloading bytes|reconstructing file|"
                   r"Fetching \d+ files|Loading (pipeline|checkpoint)|Download complete|Reconstruction complete|"
                   r"Warning: You are sending unauthenticated|^\s*\d+%\|")


def clean_stream(text, head, tail):
    lines, last = [], None
    for raw in re.split(r"[\r\n]+", ANSI.sub("", text)):
        line = raw.rstrip()
        if not line.strip() or NOISE.search(line) or line == last:
            continue
        lines.append(line)
        last = line
    if len(lines) > head + tail:
        lines = lines[:head] + [f"... [{len(lines) - head - tail} lines omitted by clean_notebook.py] ..."] + lines[-tail:]
    return "\n".join(lines) + ("\n" if lines else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--head", type=int, default=15); ap.add_argument("--tail", type=int, default=30)
    ap.add_argument("--max-image-kb", type=int, default=600)
    ap.add_argument("--strip-all", action="store_true")
    a = ap.parse_args()
    nb = json.loads(Path(a.src).read_text())
    nb.get("metadata", {}).pop("widgets", None)
    dropped = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        if a.strip_all:
            c["outputs"], c["execution_count"] = [], None
            continue
        keep = []
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                t = clean_stream("".join(o.get("text", [])), a.head, a.tail)
                if t:
                    o["text"] = t.splitlines(keepends=True); keep.append(o)
            elif o.get("output_type") == "error":
                o["traceback"] = [ANSI.sub("", x) for x in o.get("traceback", [])][-6:]; keep.append(o)
            else:
                data = o.get("data", {})
                big = [k for k, v in data.items() if k.startswith("image/") and len("".join(v)) // 1024 * 3 // 4 > a.max_image_kb]
                for k in big:
                    del data[k]; dropped += 1
                    data.setdefault("text/plain", ["<image removed: larger than %d KB, see results/>" % a.max_image_kb])
                if data:
                    keep.append(o)
        c["outputs"] = keep
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    Path(a.dst).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    s, d = Path(a.src).stat().st_size, Path(a.dst).stat().st_size
    print(f"{a.src}: {s/1e6:.2f} MB -> {a.dst}: {d/1e6:.2f} MB  ({dropped} large images removed)")


if __name__ == "__main__":
    sys.exit(main())
