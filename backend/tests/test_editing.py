"""M7: diffusion editing stage, on CPU with the fake editor (the real models are benchmarked on Kaggle)."""
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from PIL import Image  # noqa: E402

from editing import composite as C  # noqa: E402
from editing import geometry as G  # noqa: E402
from editing import run_edit  # noqa: E402
from editing.editors import FakeEditor, PrecisionError, check_precision, make_editor  # noqa: E402

W, H = 960, 540


def make_project(root: Path, n=6, absent=(4,), eid="e1"):
    """Frames with a 'sofa' rectangle moving right; Person A-style masks + index.json."""
    rng = np.random.default_rng(0)
    proj = root / "p1"
    img_dir = proj / "frames" / "images"
    mdir = proj / "edits" / eid / "masks"
    img_dir.mkdir(parents=True)
    mdir.mkdir(parents=True)
    frames = []
    for i in range(n):
        name = f"frame_{i:06d}.jpg"
        img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        x0 = 200 + 40 * i
        img[200:380, x0:x0 + 260] = (150, 60, 40)
        Image.fromarray(img).save(img_dir / name, quality=90)
        present = i not in absent
        if present:
            m = np.zeros((H, W), np.uint8)
            m[200:380, x0:x0 + 260] = 255
            Image.fromarray(m).save(mdir / f"frame_{i:06d}.png")
        frames.append({"name": name, "present": present, "source": "propagated",
                       "score": 0.9, "area_px": 180 * 260 if present else 0})
    frames[2]["area_px"] += 1  # largest -> default reference when no ref_frame is given
    (mdir / "index.json").write_text(json.dumps({"ref_frame": None, "method": "test", "frames": frames}))
    return proj


# ---------------------------------------------------------------------------------------------------- geometry
def test_model_size_multiple_of_32_and_aspect():
    for w, h, ls in [(300, 180, 768), (961, 541, 1024), (40, 700, 512), (96, 96, 768)]:
        mw, mh = G.model_size(w, h, ls)
        assert mw % 32 == 0 and mh % 32 == 0 and max(mw, mh) <= ls + 16
        assert abs(mw / mh - w / h) / (w / h) < 0.2


def test_crop_box_contains_region_and_stays_in_image():
    m = np.zeros((H, W), bool)
    m[0:30, 900:960] = True  # touching the corner
    r = G.dilate(m, 8)
    b = G.crop_box(r, 0.15, min_side=96)
    assert 0 <= b.x0 < b.x1 <= W and 0 <= b.y0 < b.y1 <= H
    assert b.w >= 60 and b.h >= 38
    ys, xs = np.nonzero(r)
    assert xs.min() >= b.x0 and xs.max() < b.x1 and ys.min() >= b.y0 and ys.max() < b.y1


def test_mask_errors(tmp_path):
    Image.fromarray(np.zeros((H, W), np.uint8)).save(tmp_path / "empty.png")
    Image.fromarray(np.full((100, 100), 255, np.uint8)).save(tmp_path / "small.png")
    with pytest.raises(G.MaskError, match="empty"):
        G.load_mask(tmp_path / "empty.png", (W, H))
    with pytest.raises(G.MaskError, match="100x100"):
        G.load_mask(tmp_path / "small.png", (W, H))


def test_klein_box_grid():
    assert G.Box(96, 54, 480, 270).grid1000(960, 540) == [100, 100, 500, 500]


# ---------------------------------------------------------------------------------------------------- composite
@pytest.mark.parametrize("feather", [0, 6])
def test_composite_never_changes_outside_and_replaces_object(feather):
    rng = np.random.default_rng(1)
    orig = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), bool)
    mask[100:300, 300:600] = True
    region = G.dilate(mask, 8)
    box = G.crop_box(region)
    alpha = C.alpha_map(region, feather)
    edit = rng.integers(0, 255, (box.h, box.w, 3), dtype=np.uint8)  # a model that changed everything in the crop
    out = C.composite(orig, edit, box, alpha)
    assert C.changed_outside(orig, out, region) == 0
    assert np.array_equal(out[~region], orig[~region])           # bit-identical outside
    inner = mask.copy()
    inner[box.slices] &= True
    sub = out[box.slices][mask[box.slices]]
    assert np.array_equal(sub, edit[mask[box.slices]])           # the object itself is fully the edit
    assert (alpha[~region] == 0).all()


def test_composite_rejects_region_outside_box():
    orig = np.zeros((H, W, 3), np.uint8)
    region = np.zeros((H, W), bool)
    region[10:20, 10:20] = True
    region[500:510, 900:910] = True
    box = G.Box(0, 0, 50, 50)
    with pytest.raises(C.CompositeError, match="outside the crop box"):
        C.composite(orig, np.zeros((50, 50, 3), np.uint8), box, C.alpha_map(region, 0))


def test_assert_unchanged_outside_detects_leak():
    a = np.zeros((10, 10, 3), np.uint8)
    b = a.copy()
    b[0, 0] = 1
    with pytest.raises(C.CompositeError, match="1 pixels"):
        C.assert_unchanged_outside(a, b, np.zeros((10, 10), bool))


# ---------------------------------------------------------------------------------------------------- precision
def test_precision_policy():
    assert check_precision("bf16", False)["option"] == "A"
    with pytest.raises(PrecisionError, match="--allow-fp16"):
        check_precision("fp16", False)
    assert check_precision("fp16", True)["option"] == "C"
    with pytest.raises(PrecisionError):
        make_editor("klein", precision="fp16")   # refused before anything is downloaded or torch imported
    with pytest.raises(PrecisionError):
        check_precision("int4", False)


# ---------------------------------------------------------------------------------------------------- stage
def test_edit_order_outward_from_reference():
    names = ["a", "b", "c", "d", "e"]
    assert run_edit.edit_order(names, "c") == [("d", "c"), ("e", "d"), ("b", "c"), ("a", "b")]


def run(argv, editor):
    assert run_edit.main(argv, editor=editor) == 0


def test_preview_then_frames_end_to_end(tmp_path):
    proj = make_project(tmp_path)
    ed = FakeEditor()
    base = ["--project-dir", str(proj), "--edit-id", "e1", "--prompt", "dark green leather sofa", "--model", "fake"]
    run(["preview", *base, "--seeds", "0,1,2"], ed)
    cdir = proj / "edits/e1/candidates"
    cand = json.loads((cdir / "index.json").read_text())
    assert cand["ref_frame"] == "frame_000002.jpg"                 # largest mask
    assert [s["seed"] for s in cand["seeds"]] == [0, 1, 2]
    for s in cand["seeds"]:
        assert (proj / "edits/e1" / s["file"]).exists() and (proj / "edits/e1" / s["crop"]).exists()
        assert s["changed_outside_px"] == 0 and s["raw_leak_psnr"] != "inf"   # the fake model leaks; composite fixes it
    assert all(c["n_images"] == 1 for c in ed.calls)                # first frame: frame + prompt only

    ed2 = FakeEditor()
    run(["frames", *base, "--chosen-seed", "1"], ed2)
    out = json.loads((proj / "edits/e1/edited/index.json").read_text())
    names = [e["frame"] for e in out["edited"]]
    assert set(names) == {"frame_000000.jpg", "frame_000001.jpg", "frame_000002.jpg", "frame_000003.jpg",
                          "frame_000005.jpg"}                       # frame 4 is not present -> untouched
    assert all(c["n_images"] == 2 for c in ed2.calls)               # frame + reference edit
    assert all(c["seed"] == 1 for c in ed2.calls)                   # same seed for every frame
    # the reference frame's edited file is the chosen candidate
    assert np.array_equal(np.asarray(Image.open(proj / "edits/e1/edited/frame_000002.png")),
                          np.asarray(Image.open(cdir / "seed_1.png")))
    # every written frame: same size, and bit-identical to the decoded JPG outside the dilated mask
    for nm in names:
        orig = G.load_image(proj / "frames/images" / nm)
        new = np.asarray(Image.open(proj / "edits/e1/edited" / (Path(nm).stem + ".png")).convert("RGB"))
        assert new.shape == orig.shape
        region = G.dilate(G.load_mask(proj / "edits/e1/masks" / (Path(nm).stem + ".png"), (W, H)), 8)
        assert C.changed_outside(orig, new, region) == 0
        assert not np.array_equal(new[region], orig[region])
    rep = json.loads((proj / "edits/e1/edit_report.json").read_text())
    assert rep["preview"]["model"] == "fake" and rep["frames"]["summary"]["frames"] == 4
    f0 = rep["frames"]["frames"][0]
    assert {"seconds", "peak_vram_mb", "raw_leak_psnr", "delta_e_vs_reference", "box", "model_size"} <= set(f0)
    # the fake model copies the reference colour -> low drift
    assert rep["frames"]["summary"]["max_delta_e_vs_reference"] < 15
    prog = [json.loads(l) for l in (proj / "edits/e1/progress.jsonl").read_text().splitlines()]
    assert any(r["stage"] == "editing frame 4 of 4" and r["progress"] == 100 for r in prog)


def test_frames_with_previous_and_keyframes(tmp_path):
    proj = make_project(tmp_path, n=8, absent=())
    base = ["--project-dir", str(proj), "--edit-id", "e1", "--prompt", "red chair", "--model", "fake"]
    run(["preview", *base, "--seeds", "5", "--ref-frame", "frame_000000.jpg"], FakeEditor())
    ed = FakeEditor()
    run(["frames", *base, "--chosen-seed", "5", "--with-previous", "--keyframe-every", "2"], ed)
    out = json.loads((proj / "edits/e1/edited/index.json").read_text())
    assert [e["frame"] for e in out["edited"]] == ["frame_000000.jpg", "frame_000002.jpg", "frame_000004.jpg",
                                                  "frame_000006.jpg"]
    assert out["skipped_present_frames"] == ["frame_000001.jpg", "frame_000003.jpg", "frame_000005.jpg",
                                             "frame_000007.jpg"]
    # with --with-previous the neighbour is only used when it was edited (keyframes skip it here)
    assert [c["n_images"] for c in ed.calls] == [2, 2, 2]


def test_stage_failures_are_reported(tmp_path):
    proj = make_project(tmp_path)
    base = ["--project-dir", str(proj), "--edit-id", "e1", "--prompt", "x", "--model", "fake"]
    assert run_edit.main(["frames", *base, "--chosen-seed", "0"], editor=FakeEditor()) == 2   # no preview yet
    run(["preview", *base, "--seeds", "0"], FakeEditor())
    assert run_edit.main(["frames", *base, "--chosen-seed", "9"], editor=FakeEditor()) == 2   # unknown seed
    (proj / "edits/e1/masks/frame_000001.png").unlink()                                      # A's mask missing
    assert run_edit.main(["preview", *base], editor=FakeEditor()) == 2
    prog = [json.loads(l) for l in (proj / "edits/e1/progress.jsonl").read_text().splitlines()]
    fails = [r for r in prog if r["status"] == "failed"]
    assert len(fails) == 3
    assert "run 'preview' first" in fails[0]["error"] and "seed 9" in fails[1]["error"]
    assert "frame_000001.png is missing" in fails[2]["error"]


def test_ref_frame_must_be_present(tmp_path):
    proj = make_project(tmp_path, absent=(4,))
    base = ["--project-dir", str(proj), "--edit-id", "e1", "--prompt", "x", "--model", "fake"]
    assert run_edit.main(["preview", *base, "--ref-frame", "frame_000004.jpg"], editor=FakeEditor()) == 2


def test_cli_rejects_fp16_without_flag(tmp_path, capsys):
    proj = make_project(tmp_path)
    rc = run_edit.main(["preview", "--project-dir", str(proj), "--edit-id", "e1", "--prompt", "x",
                        "--model", "klein", "--precision", "fp16"])
    assert rc == 2
    prog = (proj / "edits/e1/progress.jsonl").read_text()
    assert "--allow-fp16" in prog
