"""Week 2 core (CPU): mask contract, 3D lifting with a visibility check, frames<->scene undistortion maps."""
import json

import cv2
import numpy as np

from common import lift, masks, undist
from common.config import Config


# ----------------------------------------------------------------------------- lifting with analytic geometry
def _cams(n=3, W=320, H=240, f=300.0, baseline=0.15):
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return [lift.Cam(K, np.eye(3), np.array([-baseline * i, 0, 0.0]), W, H) for i in range(n)]   # camera moves +x


def _wall_depth(cam, z=5.0, scale=0.5):
    return np.full((int(cam.height * scale), int(cam.width * scale)), z, np.float32)


def _square(W, H, cx, cy, s):
    m = np.zeros((H, W), bool)
    m[cy - s:cy + s, cx - s:cx + s] = True
    return m


def test_lift_follows_parallax_on_a_wall():
    cams = _cams()
    ref = _square(320, 240, 160, 120, 30)
    pts = lift.unproject_mask(ref, _wall_depth(cams[0]), cams[0])
    assert len(pts) > 500 and np.allclose(pts[:, 2], 5.0)
    out = lift.lift_to_frame(pts, cams[2], _wall_depth(cams[2]), (240, 320))
    # camera moved 0.30 m in +x -> scene shifts left by f*0.30/5 = 18 px
    expect = _square(320, 240, 160 - 18, 120, 26)      # the lift erodes the mask by 2 depth px (SAM refines the edge later)
    assert masks.iou(out["mask"], expect) > 0.85
    assert out["visible_frac"] > 0.95


def test_occluder_in_front_hides_the_object():
    cams = _cams()
    ref = _square(320, 240, 160, 120, 30)
    pts = lift.unproject_mask(ref, _wall_depth(cams[0]), cams[0])
    d = _wall_depth(cams[2])
    d[:, :] = 2.0                                       # a surface at 2 m covers the whole view in frame 2
    out = lift.lift_to_frame(pts, cams[2], d, (240, 320))
    assert out["visible_frac"] < 0.05 and out["in_view_frac"] > 0.9 and not out["mask"].any()
    half = _wall_depth(cams[2]); half[:, :70] = 2.0     # left part of the view occluded only
    out = lift.lift_to_frame(pts, cams[2], half, (240, 320))
    assert 0.0 < out["visible_frac"] < out["in_view_frac"]


def test_object_leaving_the_view_is_reported():
    cams = _cams(n=2, baseline=6.0)                     # 6 m sideways: the object is out of frame
    ref = _square(320, 240, 160, 120, 30)
    pts = lift.unproject_mask(ref, _wall_depth(cams[0]), cams[0])
    out = lift.lift_to_frame(pts, cams[1], _wall_depth(cams[1]), (240, 320))
    assert out["in_view_frac"] == 0.0 and not out["mask"].any()


def test_prompts_from_mask_inside_and_outside():
    m = _square(320, 240, 160, 120, 40)
    p = lift.prompts_from_mask(m)
    assert all(m[y, x] for x, y in p["pos"]) and not any(m[y, x] for x, y in p["neg"])
    assert list(p["box"]) == [120.0, 80.0, 199.0, 159.0]
    assert lift.prompts_from_mask(np.zeros((10, 10), bool)) is None


# ----------------------------------------------------------------------------- undistortion maps
def test_frames_scene_maps_are_inverse():
    raw = {"model": "SIMPLE_RADIAL", "width": 640, "height": 480, "params": [500.0, 320.0, 240.0, 0.06]}
    scene = {"width": 600, "height": 450, "params": [500.0, 500.0, 300.0, 225.0]}
    s2f, f2s = undist.scene_to_frames_map(raw, scene), undist.frames_to_scene_map(raw, scene)
    # a scene pixel -> frame pixel -> back to the scene pixel
    ys, xs = np.mgrid[100:350:25, 100:500:25]
    fx, fy = s2f[0][ys, xs], s2f[1][ys, xs]
    bx = cv2.remap(f2s[0], s2f[0], s2f[1], cv2.INTER_LINEAR)       # f2s evaluated at the frame position of each scene px
    by = cv2.remap(f2s[1], s2f[0], s2f[1], cv2.INTER_LINEAR)
    assert np.abs(bx[ys, xs] - xs).max() < 0.5 and np.abs(by[ys, xs] - ys).max() < 0.5


def test_mask_roundtrip_between_spaces():
    raw = {"model": "SIMPLE_RADIAL", "width": 640, "height": 480, "params": [500.0, 320.0, 240.0, 0.05]}
    scene = {"width": 620, "height": 464, "params": [500.0, 500.0, 310.0, 232.0]}
    frame_mask = (_square(640, 480, 320, 240, 80) * 255).astype(np.uint8)
    in_scene = undist.remap(frame_mask, undist.scene_to_frames_map(raw, scene), nearest=True)
    back = undist.remap(in_scene, undist.frames_to_scene_map(raw, scene), nearest=True)
    assert masks.iou(back > 0, frame_mask > 0) > 0.93


# ----------------------------------------------------------------------------- mask contract
def _project(tmp_path, n=4):
    cfg = Config(root=str(tmp_path), project_id="p"); cfg.ensure()
    for i in range(n):
        cv2.imwrite(str(cfg.images_dir / f"{i + 1:06d}.jpg"), np.zeros((40, 60, 3), np.uint8))
    return cfg


def test_validate_catches_contract_breaks(tmp_path):
    cfg = _project(tmp_path)
    md = masks.edit_dir(cfg.project, "e1") / "masks"
    names = masks.frame_names(cfg.project)
    a = masks.save_mask(md / "000001.png", _square(60, 40, 30, 20, 8))
    idx = {"ref_frame": "000001.jpg", "method": "t", "frames": [masks.entry("000001.jpg", True, "click", 0.9, a)]}
    masks.write_index(md / "index.json", idx, names)
    assert masks.validate(cfg.project, "e1") == []
    # wrong size
    masks.save_mask(md / "000001.png", np.ones((10, 10), bool))
    assert any("size" in p for p in masks.validate(cfg.project, "e1"))
    # present without a file / file without present
    masks.save_mask(md / "000001.png", _square(60, 40, 30, 20, 8))
    masks.save_mask(md / "000002.png", _square(60, 40, 30, 20, 8))
    assert any("not present but" in p for p in masks.validate(cfg.project, "e1"))
    j = json.loads((md / "index.json").read_text())
    j["frames"][2].update(present=True, source="propagated")
    (md / "index.json").write_text(json.dumps(j))
    assert any("000003.jpg: present but" in p for p in masks.validate(cfg.project, "e1"))


def test_clean_mask_removes_specks_and_fills_holes():
    m = _square(200, 200, 100, 100, 40)
    m[96:99, 96:99] = False                        # small hole (9 px)
    m[5:7, 5:7] = True                           # speck
    c = masks.clean_mask(m)
    assert c[97, 97] and not c[5, 5] and c[100, 100]


# ----------------------------------------------------------------------------- tracking logic with injected models
def _tracking_project(tmp_path, n=6, baseline=0.15, W=320, H=240, f=300.0):
    from common.colmap_io import Camera, Image, Points3D, SparseModel, write_model
    cfg = Config(root=str(tmp_path), project_id="p"); cfg.ensure()
    cams = _cams(n=n, W=W, H=H, f=f, baseline=baseline)
    imgs = {}
    for i, c in enumerate(cams):
        name = f"{i + 1:06d}.jpg"
        cv2.imwrite(str(cfg.images_dir / name), np.zeros((H, W, 3), np.uint8))
        imgs[i + 1] = Image(i + 1, np.array([1.0, 0, 0, 0]), c.t, 1, name)
        np.save(cfg.p("week2", "depth") if False else _mk(cfg.p("week2", "depth")) / f"{i + 1:06d}.npy",
                np.full((H // 2, W // 2), 5.0, np.float16))
    model = SparseModel({1: Camera(1, "PINHOLE", W, H, np.array([f, f, W / 2, H / 2]))}, imgs,
                        Points3D(np.zeros((1, 3)), np.zeros((1, 3), np.uint8), error=np.zeros(1)))
    write_model(model, cfg.scene_dir / "sparse" / "0")
    inputs = {"images": {"same_pixels": True}, "depth": {"kind": "splat_rendered", "dir": "week2/depth"}}
    return cfg, cams, inputs


def _mk(p):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _truth(cams, j, s=26):
    shift = cams[0].K[0, 0] * (cams[j].t[0] - cams[0].t[0]) / 5.0          # px shift of a wall point
    return _square(320, 240, int(round(160 + shift)), 120, s)


def test_fused_tracking_prefers_geometry_over_a_drifting_sam2(tmp_path):
    import w2_track
    cfg, cams, inputs = _tracking_project(tmp_path, n=6)
    names = masks.frame_names(cfg.project)
    md = masks.edit_dir(cfg.project, "e1") / "masks"
    a = masks.save_mask(md / "000001.png", _square(320, 240, 160, 120, 30))
    masks.write_index(md / "index.json", {"ref_frame": "000001.jpg", "method": "click",
                                          "frames": [masks.entry("000001.jpg", True, "click", 0.97, a)]}, names)
    decoy = _square(320, 240, 40, 200, 25)

    def fake_sam2(anchors, start, cb):         # good for frames 1-3, then drifts onto a look-alike with high confidence
        return {i: ((_truth(cams, i, 30) if i <= 2 else decoy), 0.9) for i in range(6) if i not in anchors}

    def fake_refiner(rgb, pr):                 # a perfect SAM: the filled box of the prompt
        x0, y0, x1, y1 = (int(v) for v in pr["box"])
        m = np.zeros(rgb.shape[:2], bool); m[y0:y1 + 1, x0:x1 + 1] = True
        return m, 0.95

    st = w2_track.run_tracking(cfg, "e1", inputs, tracker="both", video_tracker=fake_sam2, refiner=fake_refiner)
    idx = masks.read_index(md / "index.json")
    src = {f["name"]: f["source"] for f in idx["frames"]}
    assert src["000001.jpg"] == "click" and src["000002.jpg"] == "propagated"
    assert all(src[n] == "lifted" for n in ("000004.jpg", "000005.jpg", "000006.jpg")), src       # decoy rejected
    for j in (3, 4, 5):                                                                            # and the result is right
        got = masks.load_mask(md / f"{j + 1:06d}.png")
        assert masks.iou(got, _truth(cams, j)) > 0.8, j
    assert masks.validate(cfg.project, "e1") == []
    # SAM 2 alone follows the decoy (that is exactly what the evaluation table is for)
    st2 = w2_track.run_tracking(cfg, "e1", inputs, tracker="sam2", video_tracker=fake_sam2, out_sub="track_eval/sam2/masks")
    got = masks.load_mask(masks.edit_dir(cfg.project, "e1") / "track_eval/sam2/masks" / "000005.png")
    assert masks.iou(got, _truth(cams, 4)) < 0.1 and st2["by_source"] == {"click": 1, "propagated": 5}


def test_object_out_of_view_is_not_present(tmp_path):
    import w2_track
    cfg, cams, inputs = _tracking_project(tmp_path, n=3, baseline=8.0)         # frames 2,3 look 8-16 m sideways
    names = masks.frame_names(cfg.project)
    md = masks.edit_dir(cfg.project, "e1") / "masks"
    a = masks.save_mask(md / "000001.png", _square(320, 240, 160, 120, 30))
    masks.write_index(md / "index.json", {"ref_frame": "000001.jpg", "method": "click",
                                          "frames": [masks.entry("000001.jpg", True, "click", 0.97, a)]}, names)
    sam2 = lambda anchors, start, cb: {i: (_square(320, 240, 100, 100, 20), 0.95) for i in (1, 2)}   # hallucination
    w2_track.run_tracking(cfg, "e1", inputs, tracker="both", video_tracker=sam2, refiner=None)
    idx = masks.read_index(md / "index.json")
    assert [f["present"] for f in idx["frames"]] == [True, False, False]
    assert masks.validate(cfg.project, "e1") == []


def test_user_marked_frames_are_anchors_and_kept(tmp_path):
    import w2_track
    cfg, cams, inputs = _tracking_project(tmp_path, n=4)
    names = masks.frame_names(cfg.project)
    md = masks.edit_dir(cfg.project, "e1") / "masks"
    user = _square(320, 240, 200, 150, 12)
    ents = [masks.entry("000001.jpg", True, "click", 0.97, masks.save_mask(md / "000001.png", _square(320, 240, 160, 120, 30))),
            masks.entry("000003.jpg", True, "user", 0.9, masks.save_mask(md / "000003.png", user))]
    masks.write_index(md / "index.json", {"ref_frame": "000001.jpg", "method": "click", "frames": ents}, names)
    seen = {}
    def vid(anchors, start, cb):
        seen["anchors"] = sorted(anchors)
        return {1: (_truth(cams, 1, 30), 0.9), 3: (_truth(cams, 3, 30), 0.9)}
    w2_track.run_tracking(cfg, "e1", inputs, tracker="sam2", video_tracker=vid)
    assert seen["anchors"] == [0, 2]
    assert masks.iou(masks.load_mask(md / "000003.png"), user) == 1.0            # untouched
    assert {f["name"]: f["source"] for f in masks.read_index(md / "index.json")["frames"]}["000003.jpg"] == "user"


# ----------------------------------------------------------------------------- rebuild scene (CPU part of w2_rebuild)
def _rebuild_project(tmp_path, n=4, edit_frames=("000002",)):
    from common.colmap_io import Camera, Image, Points3D, SparseModel, write_model
    cfg = Config(root=str(tmp_path), project_id="p"); cfg.ensure()
    rng = np.random.default_rng(0)
    (cfg.scene_dir / "images").mkdir(parents=True)
    imgs = {}
    for i in range(n):
        nm = f"{i + 1:06d}.jpg"
        a = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(cfg.images_dir / nm), a, [cv2.IMWRITE_JPEG_QUALITY, 95])
        shutil_copy = cfg.images_dir / nm
        (cfg.scene_dir / "images" / nm).write_bytes(shutil_copy.read_bytes())
        imgs[i + 1] = Image(i + 1, np.array([1.0, 0, 0, 0]), np.zeros(3), 1, nm)
    write_model(SparseModel({1: Camera(1, "PINHOLE", 80, 60, np.array([70.0, 70, 40, 30]))}, imgs,
                            Points3D(np.zeros((1, 3)), np.zeros((1, 3), np.uint8), error=np.zeros(1))),
                cfg.scene_dir / "sparse" / "0")
    E = masks.edit_dir(cfg.project, "e1")
    (E / "edited").mkdir(parents=True)
    items = []
    for st in edit_frames:
        orig = cv2.imread(str(cfg.scene_dir / "images" / f"{st}.jpg"))
        ed = orig.copy()
        mk = _square(80, 60, 40, 30, 10)
        ed[mk] = (0, 0, 255)                                        # "edit": paint the object red
        cv2.imwrite(str(E / "edited" / f"{st}.png"), ed)
        masks.save_mask(E / "masks" / f"{st}.png", mk)
        items.append({"name": f"{st}.jpg", "file": f"edited/{st}.png"})
    (E / "edited" / "index.json").write_text(json.dumps({"edited": items}))
    return cfg


def test_build_rebuild_scene_swaps_only_edited_frames(tmp_path):
    import w2_rebuild
    cfg = _rebuild_project(tmp_path)
    st = w2_rebuild.build_rebuild_scene(cfg, "e1", {"images": {"same_pixels": True}})
    d = masks.edit_dir(cfg.project, "e1") / "scene"
    assert st["edited_frames"] == 1
    assert (d / "images" / "000001.jpg").is_symlink() and not (d / "images" / "000002.jpg").is_symlink()
    assert (d / "sparse" / "0" / "cameras.bin").exists() or (d / "sparse" / "0" / "cameras.txt").exists()
    new = cv2.imread(str(d / "images" / "000002.jpg"))
    assert new[30, 40].tolist()[2] > 200 and new[30, 40].tolist()[0] < 60       # the edit is in the JPEG
    assert st["outside_mask_max_abs_diff"] <= 8                                   # q100 re-encode: tiny change outside
    assert (cfg.scene_dir / "images" / "000002.jpg").read_bytes() != (d / "images" / "000002.jpg").read_bytes()
    assert cv2.imread(str(cfg.scene_dir / "images" / "000002.jpg")).tolist() != new.tolist()   # original untouched


def test_rebuild_rejects_wrong_size_edit(tmp_path):
    import pytest
    import w2_rebuild
    cfg = _rebuild_project(tmp_path)
    E = masks.edit_dir(cfg.project, "e1")
    cv2.imwrite(str(E / "edited" / "000002.png"), np.zeros((30, 40, 3), np.uint8))
    with pytest.raises(RuntimeError, match="edited frame"):
        w2_rebuild.build_rebuild_scene(cfg, "e1", {"images": {"same_pixels": True}})


def test_rebuild_edits_in_frames_space_are_warped_to_scene_space(tmp_path):
    import w2_rebuild
    cfg = _rebuild_project(tmp_path)
    raw = {"model": "SIMPLE_RADIAL", "width": 80, "height": 60, "params": [70.0, 40.0, 30.0, 0.0]}   # k=0 -> identity-ish
    inputs = {"images": {"same_pixels": False, "raw_camera": raw},
              "poses": {"camera": {"model": "PINHOLE", "width": 80, "height": 60, "params": [70.0, 70.0, 40.0, 30.0]}}}
    st = w2_rebuild.build_rebuild_scene(cfg, "e1", inputs)
    new = cv2.imread(str(masks.edit_dir(cfg.project, "e1") / "scene" / "images" / "000002.jpg"))
    assert new[30, 40].tolist()[2] > 200 and st["edited_frames"] == 1


# ----------------------------------------------------------------------------- eval / test-edit / preview tools
def _toy_edit(tmp_path):
    cfg = Config(root=str(tmp_path), project_id="p"); cfg.ensure()
    for i in range(4):
        cv2.imwrite(str(cfg.images_dir / f"{i + 1:06d}.jpg"), np.full((60, 80, 3), 90, np.uint8))
    names = masks.frame_names(cfg.project)
    E = masks.edit_dir(cfg.project, "e1")
    truth = _square(80, 60, 40, 30, 10)
    # tracker A: frames 1,2 right; frame 3 wrong place; frame 4 says present though the object is absent
    ents = []
    for i, (n, mk) in enumerate(zip(names, [truth, truth, _square(80, 60, 10, 10, 6), _square(80, 60, 40, 30, 5)])):
        ents.append(masks.entry(n, True, "propagated", 0.9, masks.save_mask(E / "masks" / f"{Path(n).stem}.png", mk)))
    masks.write_index(E / "masks" / "index.json", {"ref_frame": names[0], "method": "fused", "frames": ents}, names)
    return cfg, E, names, truth


from pathlib import Path  # noqa: E402


def test_tracking_evaluation_metrics(tmp_path):
    import w2_eval_tracking as ev
    cfg, E, names, truth = _toy_edit(tmp_path)
    gt = {names[0]: truth, names[1]: truth, names[2]: truth, names[3]: None}
    r = ev.evaluate(E, ["both"], gt)["both"]
    assert r["present_ok"] == 3 and r["absent_false"] == 1 and r["presence_accuracy"] == 0.75
    assert abs(r["mean_iou_present"] - (1 + 1 + 0) / 3) < 1e-3
    assert "| both |" in ev.to_markdown({"both": r}, {"edit_id": "e1", "n_gt": 4, "n_absent": 1})


def test_make_test_edit_changes_only_inside_mask(tmp_path):
    import w2_make_test_edit as te
    cfg, E, names, truth = _toy_edit(tmp_path)
    assert te.make(cfg.project, "e1") == 4
    ed = cv2.imread(str(E / "edited" / "000001.png"))
    orig = cv2.imread(str(cfg.images_dir / "000001.jpg"))
    assert (ed[~truth] == orig[~truth]).all() and (ed[truth] != orig[truth]).any()
    assert json.loads((E / "edited" / "index.json").read_text())["edited"][0] == {"name": "000001.jpg", "file": "edited/000001.png"}


def test_preview_sheet_shape(tmp_path):
    import w2_preview as pv
    cfg, E, names, truth = _toy_edit(tmp_path)
    s = pv.sheet(cfg.project, "e1", cols=2, thumb_w=80)
    assert s.shape == (2 * 60, 2 * 80, 3)


# ----------------------------------------------------------------------------- hand-off bundle
def test_handoff_zip_layout_and_summary(tmp_path):
    import zipfile
    import make_handoff as mh
    cfg, E, names, truth = _toy_edit(tmp_path)
    (cfg.scene_dir / "sparse" / "0").mkdir(parents=True)
    (cfg.scene_dir / "sparse" / "0" / "cameras.bin").write_bytes(b"x")
    (cfg.p("week2")).mkdir()
    (cfg.p("week2") / "inputs.json").write_text(json.dumps({"total_frames": 4, "registered_frames": 4,
                                                            "required_files": ["frames", "scene/sparse", "week2/inputs.json"]}))
    (cfg.export_dir).mkdir(exist_ok=True)
    (cfg.export_dir / "scene.splat").write_bytes(b"0" * 64)
    (E / "metrics").mkdir(exist_ok=True)
    for mode, secs in (("scratch", 900.0), ("finetune", 120.0)):
        (E / "rebuilds" / mode).mkdir(parents=True)
        (E / "rebuilds" / mode / "report.json").write_text(json.dumps({
            "mode": mode, "iters": 7000, "seconds": {"train": secs, "total": secs + 20}, "num_gaussians": 1000,
            "heldout_edited": {"psnr": 25.5, "ssim": 0.88}, "heldout_original": {"psnr": 26.0},
            "jpeg_reencode": {"outside_mask_max_abs_diff": 1, "outside_mask_mean_abs_diff": 0.01}}))
    out = tmp_path / "h.zip"
    files = mh.pack(cfg, out, include_ckpt=False)
    with zipfile.ZipFile(out) as z:
        n = z.namelist()
        assert "p/HANDOFF/SUMMARY.md" in n and "p/HANDOFF/README_FOR_B.md" in n and "p/week2/inputs.json" in n
        assert "p/edits/e1/masks/index.json" in n and "p/export/scene.splat" in n and "p/frames/images/000001.jpg" in n
        summ = z.read("p/HANDOFF/SUMMARY.md").decode()
    assert "scratch" in summ and "finetune" in summ and "900 s" in summ and "at most 1 levels" in summ
    assert all(not x.startswith("/") for x in n)


def test_rebuild_main_wiring_with_faked_trainer(tmp_path, monkeypatch):
    """w2_rebuild.main end to end on CPU: s4/s5 are replaced by fakes that write what the real ones write."""
    import w2_rebuild
    cfg = _rebuild_project(tmp_path)
    cfg.gs_dir.mkdir(exist_ok=True)
    (cfg.gs_dir / "latest.json").write_text(json.dumps({"ckpt": "gs/x/ckpt.pt", "heldout": {"psnr": 26.0, "ssim": 0.9}}))
    (cfg.p("week2")).mkdir(exist_ok=True)
    (cfg.p("week2") / "inputs.json").write_text(json.dumps({"images": {"same_pixels": True}, "heldout": {"every": 8}}))
    calls = []

    def fake_sh(cmd, env=None):
        calls.append([str(c) for c in cmd])
        root, proj = cmd[cmd.index("--root") + 1], cmd[cmd.index("--project") + 1]
        ec = Config(root=root, project_id=proj)
        if "s4_train_3dgs.py" in cmd[1]:
            ec.gs_dir.mkdir(parents=True, exist_ok=True)
            (ec.gs_dir / "latest.json").write_text(json.dumps({"trainer": "gsplat", "num_gaussians": 123,
                                                               "heldout": {"psnr": 25.0, "ssim": 0.85}}))
        else:
            ec.export_dir.mkdir(parents=True, exist_ok=True)
            (ec.export_dir / "scene.splat").write_bytes(b"0" * 32)
            (ec.export_dir / "scene_meta.json").write_text(json.dumps({"num_gaussians": 123}))
            (ec.export_dir / "cameras.json").write_text("[]")

    monkeypatch.setattr(w2_rebuild, "sh", fake_sh)
    w2_rebuild.main(["--project-dir", str(cfg.project), "--edit-id", "e1", "--mode", "scratch", "--iters", "100"])
    E = masks.edit_dir(cfg.project, "e1")
    rep = json.loads((E / "rebuilds" / "scratch" / "report.json").read_text())
    assert rep["mode"] == "scratch" and rep["iters"] == 100 and rep["num_gaussians"] == 123
    assert rep["heldout_edited"]["psnr"] == 25.0 and rep["heldout_original"]["psnr"] == 26.0
    assert rep["edited_frames"] == 1 and (E / "export" / "scene.splat").exists()
    assert (E / "rebuilds" / "scratch" / "export" / "scene.splat").exists() and (E / "metrics" / "s4_rebuild.json").exists()
    s4 = calls[0]
    assert s4[s4.index("--root") + 1] == str(cfg.project / "edits") and s4[s4.index("--project") + 1] == "e1"
    assert "--iters" in s4 and s4[s4.index("--iters") + 1] == "100" and "--holdout-every" in s4
    assert not (cfg.export_dir / "scene.splat").exists()                      # the original export is untouched


def test_rebuild_finetune_without_checkpoint_fails_clearly(tmp_path):
    import pytest
    import w2_rebuild
    cfg = _rebuild_project(tmp_path)
    with pytest.raises(SystemExit, match="checkpoint"):
        w2_rebuild.main(["--project-dir", str(cfg.project), "--edit-id", "e1", "--mode", "finetune"])


def test_segment_frames_writes_contract_and_keeps_other_frames(tmp_path):
    import w2_segment
    cfg = _project(tmp_path, n=3)

    def fake_sam(rgb, pts):                         # a disc of radius 10 around the first positive click
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        x, y, _ = next(p for p in pts if p[2] == 1)
        m = (xx - x) ** 2 + (yy - y) ** 2 <= 100
        for nx, ny, lab in pts:
            if lab == 0:
                m &= ~((xx - nx) ** 2 + (yy - ny) ** 2 <= 36)
        return m, 0.93

    ref = w2_segment.segment_frames(cfg, "e1", [{"name": "000001.jpg", "clicks": [{"x": 30, "y": 20}]}], "click", fake_sam,
                                    set_ref="000001.jpg")
    assert ref["ref_frame"] == "000001.jpg" and [f["present"] for f in ref["frames"]] == [True, False, False]
    # fallback: user marks frame 3; frame 1 must stay untouched
    idx = w2_segment.segment_frames(cfg, "e1", [{"name": "000003.jpg", "clicks": [{"x": 10, "y": 10}]}], "user", fake_sam)
    by = {f["name"]: f for f in idx["frames"]}
    assert by["000001.jpg"]["source"] == "click" and by["000003.jpg"]["source"] == "user" and not by["000002.jpg"]["present"]
    assert masks.validate(cfg.project, "e1") == []
    import pytest
    with pytest.raises(ValueError, match="positive"):
        w2_segment.segment_frames(cfg, "e1", [{"name": "000001.jpg", "clicks": [{"x": 3, "y": 3, "positive": False}]}], "click", fake_sam)
    with pytest.raises(ValueError, match="unknown frame"):
        w2_segment.segment_frames(cfg, "e1", [{"name": "nope.jpg", "clicks": [{"x": 3, "y": 3}]}], "click", fake_sam)
