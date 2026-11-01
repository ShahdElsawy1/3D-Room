import json
import subprocess

import cv2
import pytest

import s1_extract_frames as s1


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    d = tmp_path_factory.mktemp("vid")
    out = d / "room.mp4"
    # 12 s 1280x720 moving test pattern; seconds 4-6 are heavily blurred (simulated fast pan)
    vf = "boxblur=luma_radius=18:luma_power=3:enable='between(t,4,6)'"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=1280x720:rate=30", "-t", "12", "-vf", vf,
                        "-pix_fmt", "yuv420p", str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("ffmpeg lavfi test source unavailable: " + r.stderr[-200:])
    return out


def run(tmp_path, video, *extra):
    s1.main(["--root", str(tmp_path), "--project", "t", "--video", str(video), *extra])
    idx = json.loads((tmp_path / "t" / "frames" / "frames_index.json").read_text())
    return idx, tmp_path / "t" / "frames" / "images"


def test_extract_resize_blur_and_index(tmp_path, video):
    idx, images = run(tmp_path, video, "--fps", "3", "--long-side", "800")
    files = sorted(images.glob("*.jpg"))
    assert len(files) == idx["kept"] and idx["extracted"] == 36
    assert idx["dropped_blurry"] >= 4                      # the blurred seconds are mostly removed
    assert files[0].name == "000001.jpg"
    h, w = cv2.imread(str(files[0])).shape[:2]
    assert max(h, w) == 800 and h % 2 == 0 and w % 2 == 0   # long side capped, even dims, aspect kept
    assert abs(w / h - 1280 / 720) < 0.01
    # dropped frames must really come from the blurred window (4-6 s)
    kept_t = [f["source_time_s"] for f in idx["frames"]]
    assert sum(1 for t in kept_t if 4.3 < t < 5.7) < 3
    assert (tmp_path / "t" / "video" / "input.mp4").exists()
    assert (tmp_path / "t" / "metrics" / "s1_extract_frames.json").exists()


def test_max_frames_cap_and_never_upscales(tmp_path, video):
    idx, images = run(tmp_path, video, "--fps", "4", "--long-side", "960", "--max-frames", "20")
    assert idx["kept"] == 20 == len(list(images.glob("*.jpg")))
    idx2, _ = run(tmp_path, video, "--fps", "2", "--long-side", "960")
    assert idx2["image_size_wh"][0] == 960


def test_auto_fps_in_spec_range(tmp_path, video):
    idx, _ = run(tmp_path, video, "--fps", "auto")
    assert 1 <= idx["fps"] <= 4 and idx["fps_selection"]["mode"] == "auto"
