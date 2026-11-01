#!/usr/bin/env python3
"""Stage 1 - frame extraction (PDF: ffmpeg, 1-4 fps, long side 640-960 px, blur filter).

    python stages/s1_extract_frames.py --project demo --video my_room.mp4 [--fps auto|1..4]

Steps
  1. ffprobe the video (duration, size, fps).
  2. Pick the sampling fps. "auto" probes motion with cheap optical flow at 4 fps and picks
     the lowest fps in [1,4] that keeps consecutive frames overlapping enough.
  3. ffmpeg: fps filter + resize (long side <= cfg.long_side, even dims) -> temp JPEGs.
  4. Blur filter: variance of the Laplacian; drop frames far below the median
     (never more than max_drop_fraction, so we don't tear holes in the trajectory).
  5. Cap at max_frames (uniform subsample), renumber 000001.jpg ... and write
     frames/frames_index.json (what Week 2 uses to map frame <-> video time).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import add_common_args, config_from_args, write_json  # noqa: E402
from common.metrics import StageMetrics  # noqa: E402
from common.progress import Progress  # noqa: E402

STAGE = "extracting frames"


def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames:stream_tags=rotate:stream_side_data=rotation",
         "-show_entries", "format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, check=True)
    j = json.loads(out.stdout)
    st = j["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    dur = float(st.get("duration") or j["format"].get("duration") or 0)
    w, h = int(st["width"]), int(st["height"])
    # ffmpeg auto-rotates phone videos; swap dims if the stream carries a 90/270 rotation
    rot = 0
    for sd in st.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = rot or int(st.get("tags", {}).get("rotate", 0) or 0)
    if abs(rot) % 180 == 90:
        w, h = h, w
    return {"width": w, "height": h, "fps": float(num) / float(den), "duration": dur, "rotation": rot}


def target_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    scale = min(1.0, long_side / max(w, h))     # never upscale
    nw, nh = int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2)
    return max(nw, 2), max(nh, 2)


def run_ffmpeg(video: Path, out_dir: Path, fps: float, size: tuple[int, int], quality: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    q = max(2, min(31, round(2 + (100 - quality) / 100 * 29)))   # jpg q 95 -> ffmpeg q ~3
    vf = f"fps={fps},scale={size[0]}:{size[1]}:flags=lanczos"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", vf,
           "-q:v", str(q), "-start_number", "0", str(out_dir / "raw_%06d.jpg")]
    subprocess.run(cmd, check=True)


def blur_score(path: Path) -> float:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def estimate_motion(video: Path, info: dict, probe_fps: float = 4.0, max_pairs: int = 60) -> float:
    """Median optical-flow magnitude between consecutive frames at `probe_fps`,
    as a fraction of image width. Cheap: Farneback on 160 px wide gray frames."""
    tmp = video.parent / "_probe"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        w = 160
        h = max(2, int(round(info["height"] * w / info["width"] / 2) * 2))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf",
                        f"fps={probe_fps},scale={w}:{h},format=gray", str(tmp / "p_%05d.png")], check=True)
        files = sorted(tmp.glob("p_*.png"))
        if len(files) < 3:
            return 0.0
        step = max(1, len(files) // (max_pairs + 1))
        files = files[::step]
        mags, prev = [], cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
        for f in files[1:]:
            cur = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mags.append(float(np.median(np.linalg.norm(flow, axis=2))) / w)
            prev = cur
        # `step` frames apart -> normalise back to one probe interval (1/probe_fps s)
        return float(np.median(mags)) / step
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def choose_fps(cfg, video: Path, info: dict) -> tuple[float, dict]:
    fc = cfg.frames
    if fc.fps != "auto":
        return float(min(max(float(fc.fps), fc.min_fps), fc.max_fps)), {"mode": "manual"}
    m4 = estimate_motion(video, info)        # motion per 1/4 s, fraction of width
    # displacement between kept frames at fps f  ~= m4 * (4 / f)  ->  want <= target_motion
    f = 4.0 * m4 / max(fc.target_motion, 1e-6)
    f = float(min(max(round(f), fc.min_fps), fc.max_fps))
    return f, {"mode": "auto", "motion_per_quarter_second": round(m4, 4),
               "target_motion": fc.target_motion}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--video", required=True, help="path to the room video (2-5 min recommended)")
    ap.add_argument("--fps", default="auto", help="'auto' or a number between 1 and 4")
    ap.add_argument("--long-side", type=int, default=960, help="640-960 per the spec")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--blur-rel", type=float, default=0.5)
    ap.add_argument("--blur-abs", type=float, default=15.0)
    args = ap.parse_args(argv)

    cfg = config_from_args(args)
    cfg.frames.fps, cfg.frames.long_side = args.fps, args.long_side
    cfg.frames.max_frames, cfg.frames.blur_rel, cfg.frames.blur_abs = args.max_frames, args.blur_rel, args.blur_abs
    if not 640 <= cfg.frames.long_side <= 960:
        print(f"[warn] long side {cfg.frames.long_side} is outside the 640-960 px budget in the spec", file=sys.stderr)

    prog = Progress(cfg.metrics_dir, args.backend_url, args.task_id)
    prog.start(STAGE)
    src = Path(args.video)
    if not src.exists():
        prog.fail(STAGE, f"video not found: {src}")
        raise SystemExit(f"video not found: {src}")

    with StageMetrics(cfg.metrics_dir, "s1_extract_frames") as m:
        try:
            # keep the original video inside the project (Week 2 / backend reuse it)
            dst = cfg.video_dir / f"input{src.suffix.lower()}"
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            info = probe(dst)
            prog.report(STAGE, 0.05)
            size = target_size(info["width"], info["height"], cfg.frames.long_side)
            fps, fps_info = choose_fps(cfg, dst, info)
            prog.report(STAGE, 0.15)

            raw = cfg.frames_dir / "_raw"
            shutil.rmtree(raw, ignore_errors=True)
            run_ffmpeg(dst, raw, fps, size, cfg.frames.jpg_quality)
            raw_files = sorted(raw.glob("raw_*.jpg"))
            if len(raw_files) < 10:
                raise RuntimeError(f"only {len(raw_files)} frames extracted - video too short or unreadable")
            prog.report(STAGE, 0.5)

            scores = np.array([blur_score(f) for f in raw_files])
            prog.report(STAGE, 0.8)
            thr = max(cfg.frames.blur_abs, cfg.frames.blur_rel * float(np.median(scores)))
            keep = scores >= thr
            max_drop = int(cfg.frames.max_drop_fraction * len(scores))
            if (~keep).sum() > max_drop:       # too aggressive: keep the sharpest instead
                order = np.argsort(scores)
                keep = np.ones(len(scores), bool)
                keep[order[:max_drop]] = False
            kept_idx = np.where(keep)[0]
            if len(kept_idx) > cfg.frames.max_frames:      # uniform subsample to the cap
                sel = np.linspace(0, len(kept_idx) - 1, cfg.frames.max_frames).round().astype(int)
                kept_idx = kept_idx[sel]

            # (re)write the final numbered images
            if cfg.images_dir.exists():
                shutil.rmtree(cfg.images_dir)
            cfg.images_dir.mkdir(parents=True)
            index = []
            for n, i in enumerate(kept_idx, start=1):
                name = f"{n:06d}.jpg"
                shutil.move(str(raw_files[i]), cfg.images_dir / name)
                index.append({"name": name, "source_time_s": round(float(i) / fps, 3),
                              "raw_index": int(i), "blur_score": round(float(scores[i]), 2)})
            shutil.rmtree(raw, ignore_errors=True)

            write_json(cfg.frames_index, {
                "video": str(dst), "video_info": info, "fps": fps, "fps_selection": fps_info,
                "image_size_wh": list(size), "blur_threshold": round(thr, 2),
                "extracted": int(len(scores)), "dropped_blurry": int((~keep).sum()),
                "kept": int(len(index)), "frames": index,
                "params": {"long_side": cfg.frames.long_side, "max_frames": cfg.frames.max_frames},
            })
            m.extra.update(frames_extracted=int(len(scores)), frames_kept=len(index),
                           dropped_blurry=int((~keep).sum()), fps=fps, size_wh=list(size))
            print(f"[s1] {len(scores)} frames @ {fps} fps -> kept {len(index)} "
                  f"({int((~keep).sum())} blurry dropped), size {size[0]}x{size[1]}")
            prog.finish(STAGE)
        except Exception as e:
            prog.fail(STAGE, e)
            raise


if __name__ == "__main__":
    main()
