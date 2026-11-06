"""Video validation with ffprobe (no decoding of the whole file)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


class VideoError(ValueError):
    pass


def _fps(stream: dict) -> float | None:
    for key in ("avg_frame_rate", "r_frame_rate"):
        v = stream.get(key) or ""
        if "/" in v:
            n, d = v.split("/", 1)
            try:
                if float(d) > 0 and float(n) > 0:
                    return round(float(n) / float(d), 3)
            except ValueError:
                pass
    return None


def _rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    try:
        return int((stream.get("tags") or {}).get("rotate", 0)) % 360
    except ValueError:
        return 0


def probe_video(path: Path, ffprobe_bin: str = "ffprobe") -> dict:
    """Return duration/size/fps/codec. Raises VideoError if the file is not a readable video."""
    try:
        r = subprocess.run([ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                            str(path)], capture_output=True, text=True, timeout=60)
    except FileNotFoundError as e:
        raise RuntimeError(f"ffprobe not found ('{ffprobe_bin}'); install ffmpeg or set BACKEND_FFPROBE") from e
    except subprocess.TimeoutExpired as e:
        raise VideoError("ffprobe timed out reading the file") from e
    if r.returncode != 0:
        raise VideoError("not a readable video file")
    info = json.loads(r.stdout or "{}")
    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"
                and (s.get("disposition") or {}).get("attached_pic", 0) == 0]
    if not vstreams:
        raise VideoError("file contains no video stream")
    s = vstreams[0]
    duration = None
    for src in (info.get("format") or {}, s):
        try:
            duration = float(src.get("duration"))
            break
        except (TypeError, ValueError):
            continue
    if not duration or duration <= 0:
        raise VideoError("could not determine video duration")
    w, h = int(s.get("width") or 0), int(s.get("height") or 0)
    if w <= 0 or h <= 0:
        raise VideoError("could not determine video resolution")
    rot = _rotation(s)
    if rot in (90, 270):          # report the size as displayed (ffmpeg auto-rotates during extraction)
        w, h = h, w
    return {"duration_s": round(duration, 3), "width": w, "height": h, "fps": _fps(s),
            "codec": s.get("codec_name"), "rotation": rot}


def check_rules(meta: dict, min_s: float, max_s: float, recommended: tuple[float, float]) -> list[str]:
    """Hard limits raise VideoError; soft spec guidance (2-5 min) returns warnings."""
    d = meta["duration_s"]
    if d < min_s:
        raise VideoError(f"video is {d:.1f}s; minimum is {min_s:.0f}s")
    if d > max_s:
        raise VideoError(f"video is {d:.1f}s; maximum is {max_s:.0f}s (spec: 2-5 min)")
    warnings = []
    lo, hi = recommended
    if not lo <= d <= hi:
        warnings.append(f"duration {d:.0f}s is outside the recommended {lo/60:.0f}-{hi/60:.0f} min")
    if min(meta["width"], meta["height"]) < 480:
        warnings.append(f"low resolution {meta['width']}x{meta['height']}; frames are resized to 640-960 px "
                        "on the long side, so reconstruction quality may suffer")
    return warnings
