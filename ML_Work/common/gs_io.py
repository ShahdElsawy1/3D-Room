"""3D Gaussian Splatting file formats (numpy only, torch only to open .pt checkpoints).

  * trainer .ply   -> the standard 3DGS PLY (x y z nx ny nz f_dc_* f_rest_* opacity scale_* rot_*)
  * .splat         -> the compact 32-byte-per-Gaussian format read by antimatter15/splat and
                      many three.js / web viewers   (pos f32x3, scale f32x3, rgba u8x4, quat u8x4)
  * cameras.json   -> training cameras in the Inria viewer convention (position + c2w rotation)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814


# ------------------------------------------------------------------- PLY
def read_gs_ply(path) -> dict:
    path = Path(path)
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            header.append(line)
            if line == "end_header":
                break
            if not line:
                raise ValueError("bad PLY header")
        if "binary_little_endian" not in " ".join(header):
            raise ValueError("only binary_little_endian PLY is supported")
        n = int(next(l.split()[-1] for l in header if l.startswith("element vertex")))
        props = [l.split()[-1] for l in header if l.startswith("property")]
        types = [l.split()[1] for l in header if l.startswith("property")]
        if any(t != "float" for t in types):
            raise ValueError(f"expected float32 properties, got {set(types)}")
        data = np.frombuffer(f.read(n * 4 * len(props)), dtype="<f4").reshape(n, len(props))
    return {name: data[:, i] for i, name in enumerate(props)}


def write_gs_ply(path, means, sh0, shN, opacities_logit, scales_log, quats) -> int:
    """Standard 3DGS PLY. sh0 (N,1,3), shN (N,K,3), scales are log-scale, opacities are logits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(means)
    f_dc = sh0.reshape(n, -1)                                   # (N,3)
    f_rest = shN.transpose(0, 2, 1).reshape(n, -1) if shN is not None and shN.size else np.zeros((n, 0), np.float32)
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(f_dc.shape[1])] +
             [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"] +
             [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    cols = [means, np.zeros((n, 3), np.float32), f_dc, f_rest, opacities_logit.reshape(n, 1), scales_log, quats]
    arr = np.concatenate([c.astype(np.float32) for c in cols], axis=1)
    assert arr.shape[1] == len(names)
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + \
             "".join(f"property float {nm}\n" for nm in names) + "end_header\n"
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.astype("<f4").tobytes())
    return n


def ckpt_to_ply(ckpt_path, ply_path) -> int:
    """gsplat checkpoint (.pt with ckpt['splats']) -> standard 3DGS PLY."""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    s = ck["splats"]
    g = lambda k: s[k].detach().float().cpu().numpy()  # noqa: E731
    return write_gs_ply(ply_path, g("means"), g("sh0"), g("shN") if "shN" in s else None,
                        g("opacities"), g("scales"), g("quats"))


# ------------------------------------------------------------------- .splat
def ply_to_splat(ply_path, splat_path, sort: bool = True) -> dict:
    d = read_gs_ply(ply_path)
    n = len(d["x"])
    pos = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float32)
    scale = np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], 1)).astype(np.float32)
    rgb = np.stack([0.5 + SH_C0 * d["f_dc_0"], 0.5 + SH_C0 * d["f_dc_1"], 0.5 + SH_C0 * d["f_dc_2"]], 1)
    alpha = 1.0 / (1.0 + np.exp(-d["opacity"]))
    rgba = np.concatenate([np.clip(rgb * 255, 0, 255), np.clip(alpha[:, None] * 255, 0, 255)], 1).astype(np.uint8)
    q = np.stack([d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"]], 1)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    rot = np.clip(q * 128 + 128, 0, 255).astype(np.uint8)
    if sort:   # big + opaque first so a progressive viewer shows the structure early
        order = np.argsort(-(scale.prod(1) * alpha))
        pos, scale, rgba, rot = pos[order], scale[order], rgba[order], rot[order]
    rec = np.empty(n, dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"], rec["s"], rec["c"], rec["r"] = pos, scale, rgba, rot
    Path(splat_path).parent.mkdir(parents=True, exist_ok=True)
    Path(splat_path).write_bytes(rec.tobytes())
    return {"num_gaussians": int(n), "bbox_min": pos.min(0).tolist(), "bbox_max": pos.max(0).tolist(),
            "median_position": np.median(pos, 0).tolist()}


# ------------------------------------------------------------------- cameras
def cameras_json(model, frame_index=None, poses=None) -> list:
    """Training cameras, Inria-viewer style (rotation = camera-to-world)."""
    out = []
    for im in sorted(model.images.values(), key=lambda i: i.name):
        cam = model.cameras[im.camera_id]
        K = cam.K()
        R_w2c, t = im.R, im.tvec
        if poses is not None and im.name in poses:        # poses in the splat's own space (common/gs_space.py)
            R_w2c, t = np.asarray(poses[im.name])[:3, :3], np.asarray(poses[im.name])[:3, 3]
        out.append({"id": im.id, "img_name": im.name, "width": cam.width, "height": cam.height,
                    "position": (-R_w2c.T @ t).tolist(), "rotation": R_w2c.T.tolist(),
                    "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])})
    return out
