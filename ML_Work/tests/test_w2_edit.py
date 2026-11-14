"""Diffusion edit stage (CPU): sizes, compositing guarantee, reference choice, stage wiring with a fake model, and the
hand-over to w2_rebuild. The real model (common.diffusion_edit.QwenEditor) needs a GPU and is smoke-tested on Kaggle."""
import json

import cv2
import numpy as np
import pytest

from common import diffusion_edit as D
from common import masks
from common.config import Config


def _square(W, H, cx, cy, s):
    m = np.zeros((H, W), bool)
    m[cy - s:cy + s, cx - s:cx + s] = True
    return m


# ----------------------------------------------------------------------------- sizes
@pytest.mark.parametrize("mode", ["pad", "resize"])
def test_fit_unfit_roundtrip_and_multiple_of_16(mode):
    img = (np.random.default_rng(0).random((61, 90, 3)) * 255).astype(np.uint8)
    fit, info = D.fit_to_multiple(img, mode=mode)
    assert fit.shape[0] % 16 == 0 and fit.shape[1] % 16 == 0 and fit.shape[:2] == (64, 96)
    back = D.unfit(fit, info)
    assert back.shape == img.shape
    if mode == "pad":
        assert (back == img).all()                              # pixels never move when padding


def test_unfit_accepts_a_model_output_of_another_size():
    img = np.zeros((61, 90, 3), np.uint8)
    _, info = D.fit_to_multiple(img)
    odd = np.zeros((70, 100, 3), np.uint8)                      # the pipeline rounded differently
    assert D.unfit(odd, info).shape == (61, 90, 3)
    assert D.gen_size((64, 96), 1.5) == (96, 144) and D.gen_size((64, 96), 1.0) == (64, 96)


# ----------------------------------------------------------------------------- the guarantee
def test_composite_is_bit_identical_outside_the_grown_mask():
    rng = np.random.default_rng(1)
    orig = (rng.random((80, 100, 3)) * 255).astype(np.uint8)
    edited = (rng.random((80, 100, 3)) * 255).astype(np.uint8)  # the "model" changed EVERY pixel
    mk = _square(100, 80, 50, 40, 15)
    out = D.composite(orig, edited, mk, grow=2, feather=2)
    region = D.grow_mask(mk, 2)
    assert (out[~region] == orig[~region]).all()
    deep = cv2.erode(mk.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert (out[deep] == edited[deep]).all()                    # inside the mask the edit is pure
    assert D.outside_diff(orig, out, mk, grow=2)["max_abs_diff"] == 0
    assert D.outside_diff(orig, out, mk, grow=0)["max_abs_diff"] > 0   # the 2 px rim is where the blend lives


def test_composite_hard_mask_is_exact():
    rng = np.random.default_rng(2)
    orig = (rng.random((40, 40, 3)) * 255).astype(np.uint8)
    edited = (rng.random((40, 40, 3)) * 255).astype(np.uint8)
    mk = _square(40, 40, 20, 20, 8)
    out = D.composite(orig, edited, mk, grow=0, feather=0)
    assert (out[mk] == edited[mk]).all() and (out[~mk] == orig[~mk]).all()
    with pytest.raises(ValueError):
        D.composite(orig, edited[:10], mk)


def test_outline_hint_paints_only_outside_the_mask():
    rgb = np.full((80, 100, 3), 90, np.uint8)
    mk = _square(100, 80, 50, 40, 12)
    h = D.outline_hint(rgb, mk, color=(255, 0, 0), thickness=3, gap=2)
    assert (h[mk] == rgb[mk]).all()                             # the object itself is never painted
    ring = (h != rgb).any(axis=2)
    assert ring.any() and not (ring & D.grow_mask(mk, 2)).any()  # starts beyond the grown mask: it never reaches the output


# ----------------------------------------------------------------------------- reference + prompt
def test_pick_reference_prefers_a_large_uncut_object():
    cut = np.zeros((60, 80), bool); cut[0:30, 0:30] = True       # 900 px but cut by the frame edge -> 450
    inside = _square(80, 60, 40, 30, 16)                          # 1024 px, fully visible
    small = _square(80, 60, 40, 30, 5)
    assert D.pick_reference({"a.jpg": cut, "b.jpg": inside, "c.jpg": small}) == "b.jpg"
    with pytest.raises(ValueError):
        D.pick_reference({})


def test_prompt_mentions_reference_only_when_given():
    p0 = D.build_prompt("make the sofa dark green leather.", with_reference=False, hint=True)
    p1 = D.build_prompt("make the sofa dark green leather", with_reference=True, hint=False)
    assert "Image 2" not in p0 and "red outline" in p0 and "leather.." not in p0
    assert "Image 2" in p1 and "red outline" not in p1


# ----------------------------------------------------------------------------- the stage with a fake model
class FakeEditor:
    """Returns noise everywhere (so only the compositing can keep the frame intact) and records what it was given."""

    def __init__(self):
        self.calls = []
        self.load_peak_mb = 123.0

    def peak_mb(self):
        return 456.0

    def edit(self, images, prompt, seed, size_hw):
        self.calls.append({"n": len(images), "shapes": [i.shape for i in images], "prompt": prompt, "seed": seed,
                           "size": size_hw, "images": [i.copy() for i in images]})
        rng = np.random.default_rng(seed)
        return (rng.random((*size_hw, 3)) * 255).astype(np.uint8)


def _project(tmp_path, n=5, W=84, H=60, with_scene=False):
    cfg = Config(root=str(tmp_path), project_id="p"); cfg.ensure()
    rng = np.random.default_rng(7)
    for i in range(n):
        cv2.imwrite(str(cfg.images_dir / f"{i + 1:06d}.jpg"), (rng.random((H, W, 3)) * 255).astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    names = masks.frame_names(cfg.project)
    md = masks.edit_dir(cfg.project, "e1") / "masks"
    ents = []
    for i, nm in enumerate(names[:4]):                           # frames 1-4 contain the object, frame 5 does not
        s = 8 + 2 * i
        a = masks.save_mask(md / f"{i + 1:06d}.png", _square(W, H, 40, 30, s))
        ents.append(masks.entry(nm, True, "click" if i == 0 else "propagated", 0.9, a))
    masks.write_index(md / "index.json", {"ref_frame": names[0], "method": "t", "frames": ents}, names)
    return cfg


def _run(cfg, extra, editor):
    import w2_edit
    w2_edit.main(["--project-dir", str(cfg.project), "--edit-id", "e1", "--prompt", "make the sofa dark green leather", *extra],
                 editor_factory=lambda: editor)


def test_stage_reference_then_rest_contract(tmp_path):
    cfg = _project(tmp_path)
    E = masks.edit_dir(cfg.project, "e1")
    ed = FakeEditor()
    _run(cfg, ["--phase", "reference", "--n-seeds", "3", "--seed", "5"], ed)
    ref = json.loads((E / "edit" / "reference" / "index.json").read_text())
    assert ref["ref_frame"] == "000004.jpg"                      # the largest mask
    assert [c["seed"] for c in ref["candidates"]] == [5, 6, 7] and (E / "preview_reference.png").exists()
    assert all(c["n"] == 1 and c["shapes"][0][:2] == (64, 96) for c in ed.calls)      # padded to a multiple of 16
    with pytest.raises(SystemExit, match="choose the reference edit"):
        _run(cfg, ["--phase", "rest"], FakeEditor())

    ed2 = FakeEditor()
    _run(cfg, ["--phase", "rest", "--ref-seed", "6"], ed2)
    idx = json.loads((E / "edited" / "index.json").read_text())
    assert [e["name"] for e in idx["edited"]] == ["000001.jpg", "000002.jpg", "000003.jpg", "000004.jpg"]   # not frame 5
    assert {e["name"]: e["role"] for e in idx["edited"]}["000004.jpg"] == "reference"
    assert len(ed2.calls) == 3 and all(c["n"] == 2 and c["seed"] == 6 for c in ed2.calls)   # reference as 2nd condition image
    assert "Image 2" in ed2.calls[0]["prompt"]
    ref_png = cv2.cvtColor(cv2.imread(str(E / "edited" / "000004.png")), cv2.COLOR_BGR2RGB)
    assert (ed2.calls[0]["images"][1][:60, :84] == ref_png).all()                      # the chosen edit, not another seed
    assert masks.validate(cfg.project, "e1") == [] and D.validate_edited(cfg.project, "e1") == []
    # the guarantee, read back from the files on disk
    for e in idx["edited"]:
        orig = cv2.imread(str(cfg.images_dir / e["name"]))
        new = cv2.imread(str(E / e["file"]))
        mk = masks.load_mask(E / "masks" / f"{e['name'][:-4]}.png")
        assert new.shape == orig.shape
        assert (new[~D.grow_mask(mk, 2)] == orig[~D.grow_mask(mk, 2)]).all() and (new != orig).any()
    m = json.loads((E / "metrics" / "w2_edit.json").read_text())
    assert m["outside_grown_mask_max_abs_diff"] == 0 and m["frames_edited"] == 4 and m["quant"] == "4bit"
    assert m["generation_peak_vram_mb"] == 456.0 and (E / "metrics" / "w2_edit_4bit_bfloat16_none_s40.json").exists()
    assert (E / "preview_edited.png").exists()


def test_stage_all_resume_stride_and_limit(tmp_path):
    cfg = _project(tmp_path)
    E = masks.edit_dir(cfg.project, "e1")
    ed = FakeEditor()
    _run(cfg, ["--phase", "all", "--n-seeds", "2", "--limit", "1"], ed)          # smoke: 2 candidates + 1 frame
    assert len(ed.calls) == 3
    idx = json.loads((E / "edited" / "index.json").read_text())
    assert len(idx["edited"]) == 2 and idx["ref_seed"] == 0                       # default reference = the first seed
    ed2 = FakeEditor()
    _run(cfg, ["--phase", "rest", "--ref-seed", "0", "--resume"], ed2)           # finishes the rest, keeps the first
    assert len(ed2.calls) == 2
    assert len(json.loads((E / "edited" / "index.json").read_text())["edited"]) == 4
    ed3 = FakeEditor()
    _run(cfg, ["--phase", "rest", "--ref-seed", "0", "--stride", "2"], ed3)      # keyframes: every 2nd of the 3 others
    assert len(ed3.calls) == 2 and len(json.loads((E / "edited" / "index.json").read_text())["edited"]) == 3


def test_ref_file_is_composited_so_outside_stays_original(tmp_path):
    cfg = _project(tmp_path)
    E = masks.edit_dir(cfg.project, "e1")
    junk = (np.random.default_rng(3).random((60, 84, 3)) * 255).astype(np.uint8)   # someone's edit that changed everything
    cv2.imwrite(str(tmp_path / "ref.png"), junk)
    _run(cfg, ["--phase", "rest", "--ref-frame", "000004.jpg", "--ref-file", str(tmp_path / "ref.png"), "--limit", "1"], FakeEditor())
    got = cv2.imread(str(E / "edited" / "000004.png"))
    orig = cv2.imread(str(cfg.images_dir / "000004.jpg"))
    mk = masks.load_mask(E / "masks" / "000004.png")
    assert (got[~D.grow_mask(mk, 2)] == orig[~D.grow_mask(mk, 2)]).all()
    with pytest.raises(SystemExit, match="ref-frame"):
        _run(cfg, ["--phase", "rest", "--ref-file", str(tmp_path / "ref.png")], FakeEditor())


def test_stage_needs_masks(tmp_path):
    cfg = Config(root=str(tmp_path), project_id="p"); cfg.ensure()
    cv2.imwrite(str(cfg.images_dir / "000001.jpg"), np.zeros((32, 32, 3), np.uint8))
    with pytest.raises(SystemExit, match="no masks"):
        import w2_edit
        w2_edit.main(["--project-dir", str(cfg.project), "--edit-id", "e1", "--prompt", "x"], editor_factory=FakeEditor)
    assert json.loads((masks.edit_dir(cfg.project, "e1") / "metrics" / "progress.jsonl").read_text().splitlines()[-1])["status"] == "failed"


def test_real_editor_without_diffusers_fails_with_the_fix(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "diffusers", None)          # makes `import diffusers` raise ImportError
    with pytest.raises(RuntimeError, match="setup_kaggle.sh diffusion"):
        D.QwenEditor()


# ----------------------------------------------------------------------------- hand-over to the rebuild + the table
def test_edited_frames_are_accepted_by_the_rebuild(tmp_path):
    import w2_edit
    import w2_rebuild
    from test_w2_core import _rebuild_project
    cfg = _rebuild_project(tmp_path, n=4, edit_frames=())
    E = masks.edit_dir(cfg.project, "e2")
    names = masks.frame_names(cfg.project)
    ents = []
    for i, nm in enumerate(names[:3]):
        ents.append(masks.entry(nm, True, "click" if i == 0 else "propagated", 0.9,
                                masks.save_mask(E / "masks" / f"{nm[:-4]}.png", _square(80, 60, 40, 30, 8 + i))))
    masks.write_index(E / "masks" / "index.json", {"ref_frame": names[0], "method": "t", "frames": ents}, names)
    w2_edit.main(["--project-dir", str(cfg.project), "--edit-id", "e2", "--prompt", "recolour", "--phase", "all", "--n-seeds", "1"],
                 editor_factory=FakeEditor)
    assert [i["stem"] for i in w2_rebuild.read_edited_items(E)] == ["000001", "000002", "000003"]
    st = w2_rebuild.build_rebuild_scene(cfg, "e2", {"images": {"same_pixels": True}})
    assert st["edited_frames"] == 3 and st["outside_mask_max_abs_diff"] <= 8


def test_diffusion_table_from_two_settings(tmp_path):
    import w2_edit_table as T
    E = tmp_path / "p" / "edits" / "e1"
    (E / "metrics").mkdir(parents=True)
    for tag, q, spf, vr in (("4bit_bfloat16_none_s40", "4bit", 38.2, 9100.0), ("8bit_bfloat16_none_s40", "8bit", 51.0, 13800.0)):
        (E / "metrics" / f"w2_edit_{tag}.json").write_text(json.dumps({
            "tag": tag, "model": "Qwen/Qwen-Image-2.1", "quant": q, "dtype": "bfloat16", "offload": "none", "steps": 40,
            "model_load_seconds": 70.0, "seconds_per_frame_mean": spf, "seconds_per_frame_median": spf, "peak_vram_mb": vr,
            "generation_peak_vram_mb": vr - 500, "load_peak_vram_mb": vr - 2000, "frames_edited": 20,
            "outside_grown_mask_max_abs_diff": 0, "ok": True}))
    t = T.table(E)
    assert t.count("\n") == 3 and "| 4bit_bfloat16_none_s40 |" in t and "9100 MiB" in t and "51.0 s" in t
    assert T.table(tmp_path) is None
