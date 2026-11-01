"""Read/write COLMAP sparse models (text AND binary) without needing COLMAP installed.

This is the interchange format the whole project relies on:

    <folder>/images/*.jpg
    <folder>/sparse/0/cameras.bin|txt
    <folder>/sparse/0/images.bin|txt
    <folder>/sparse/0/points3D.bin|txt

Most 3DGS trainers (original Inria code, gsplat's example parser, nerfstudio) read this.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import qvec2rotmat, rotmat2qvec

# name -> (model_id, num_params)
CAMERA_MODELS = {
    "SIMPLE_PINHOLE": (0, 3),   # f, cx, cy
    "PINHOLE": (1, 4),          # fx, fy, cx, cy
    "SIMPLE_RADIAL": (2, 4),    # f, cx, cy, k
    "RADIAL": (3, 5),           # f, cx, cy, k1, k2
    "OPENCV": (4, 8),           # fx, fy, cx, cy, k1, k2, p1, p2
}
MODEL_ID_TO_NAME = {v[0]: k for k, v in CAMERA_MODELS.items()}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def K(self) -> np.ndarray:
        p = self.params
        if self.model == "PINHOLE" or self.model == "OPENCV":
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        else:  # SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL: single focal
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    @property
    def has_distortion(self) -> bool:
        return self.model not in ("PINHOLE", "SIMPLE_PINHOLE")


@dataclass
class Image:
    id: int
    qvec: np.ndarray           # (w, x, y, z)  world -> camera
    tvec: np.ndarray
    camera_id: int
    name: str

    @property
    def R(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)


@dataclass
class Points3D:
    xyz: np.ndarray            # (N,3) float64
    rgb: np.ndarray            # (N,3) uint8
    error: np.ndarray = field(default=None)  # (N,)
    ids: np.ndarray = field(default=None)    # (N,) uint64

    def __post_init__(self):
        n = len(self.xyz)
        if self.error is None:
            self.error = np.zeros(n)
        if self.ids is None:
            self.ids = np.arange(1, n + 1, dtype=np.uint64)


@dataclass
class SparseModel:
    cameras: dict            # id -> Camera
    images: dict             # id -> Image
    points: Points3D

    def summary(self) -> dict:
        return {"num_cameras": len(self.cameras), "num_images": len(self.images),
                "num_points3D": int(len(self.points.xyz))}


# ===================================================================== writers
def write_text(model: SparseModel, folder) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "cameras.txt").open("w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(model.cameras)}\n")
        for c in model.cameras.values():
            f.write(f"{c.id} {c.model} {c.width} {c.height} " + " ".join(f"{x:.10g}" for x in c.params) + "\n")
    with (folder / "images.txt").open("w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(model.images)}\n")
        for im in model.images.values():
            q, t = im.qvec, im.tvec
            f.write(f"{im.id} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g} "
                    f"{t[0]:.12g} {t[1]:.12g} {t[2]:.12g} {im.camera_id} {im.name}\n\n")
    p = model.points
    with (folder / "points3D.txt").open("w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(p.xyz)}\n")
        for i in range(len(p.xyz)):
            x, y, z = p.xyz[i]
            r, g, b = p.rgb[i]
            f.write(f"{int(p.ids[i])} {x:.8g} {y:.8g} {z:.8g} {int(r)} {int(g)} {int(b)} {p.error[i]:.6g}\n")


def write_binary(model: SparseModel, folder) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(model.cameras)))
        for c in model.cameras.values():
            mid, n = CAMERA_MODELS[c.model]
            assert len(c.params) == n, f"{c.model} needs {n} params"
            f.write(struct.pack("<iiQQ", c.id, mid, c.width, c.height))
            f.write(struct.pack(f"<{n}d", *[float(x) for x in c.params]))
    with (folder / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(model.images)))
        for im in model.images.values():
            f.write(struct.pack("<i", im.id))
            f.write(struct.pack("<4d", *[float(x) for x in im.qvec]))
            f.write(struct.pack("<3d", *[float(x) for x in im.tvec]))
            f.write(struct.pack("<i", im.camera_id))
            f.write(im.name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # no 2D keypoints stored
    p = model.points
    with (folder / "points3D.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(p.xyz)))
        for i in range(len(p.xyz)):
            f.write(struct.pack("<Q", int(p.ids[i])))
            f.write(struct.pack("<3d", *[float(x) for x in p.xyz[i]]))
            f.write(struct.pack("<3B", *[int(x) for x in p.rgb[i]]))
            f.write(struct.pack("<d", float(p.error[i])))
            f.write(struct.pack("<Q", 0))  # empty track


def write_model(model: SparseModel, folder, text: bool = True, binary: bool = True) -> None:
    if binary:
        write_binary(model, folder)
    if text:
        write_text(model, folder)


# ===================================================================== readers
def _read_text(folder: Path) -> SparseModel:
    cams, imgs = {}, {}
    for line in (folder / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        e = line.split()
        cams[int(e[0])] = Camera(int(e[0]), e[1], int(e[2]), int(e[3]), np.array(e[4:], dtype=np.float64))
    lines = [l for l in (folder / "images.txt").read_text().splitlines()]
    body = [l for l in lines if not l.startswith("#")]
    for i in range(0, len(body) - 1, 2):
        e = body[i].split()
        if len(e) < 10:
            continue
        imgs[int(e[0])] = Image(int(e[0]), np.array(e[1:5], dtype=np.float64),
                                np.array(e[5:8], dtype=np.float64), int(e[8]), e[9])
    xyz, rgb, err, ids = [], [], [], []
    pf = folder / "points3D.txt"
    if pf.exists():
        for line in pf.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            e = line.split()
            ids.append(int(e[0])); xyz.append([float(v) for v in e[1:4]])
            rgb.append([int(v) for v in e[4:7]]); err.append(float(e[7]))
    pts = Points3D(np.array(xyz).reshape(-1, 3), np.array(rgb, dtype=np.uint8).reshape(-1, 3),
                   np.array(err), np.array(ids, dtype=np.uint64))
    return SparseModel(cams, imgs, pts)


def _read_binary(folder: Path) -> SparseModel:
    cams, imgs = {}, {}
    with (folder / "cameras.bin").open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
            name = MODEL_ID_TO_NAME.get(mid)
            if name is None:
                raise ValueError(f"unsupported COLMAP camera model id {mid} in {folder}")
            k = CAMERA_MODELS[name][1]
            params = np.array(struct.unpack(f"<{k}d", f.read(8 * k)))
            cams[cid] = Camera(cid, name, w, h, params)
    with (folder / "images.bin").open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            q = np.array(struct.unpack("<4d", f.read(32)))
            t = np.array(struct.unpack("<3d", f.read(24)))
            cid = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00" or not ch:
                    break
                name += ch
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * n2d, 1)  # (x, y, point3D_id) = 8+8+8 bytes
            imgs[iid] = Image(iid, q, t, cid, name.decode())
    xyz, rgb, err, ids = [], [], [], []
    pf = folder / "points3D.bin"
    if pf.exists():
        with pf.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                pid = struct.unpack("<Q", f.read(8))[0]
                xyz.append(struct.unpack("<3d", f.read(24)))
                rgb.append(struct.unpack("<3B", f.read(3)))
                err.append(struct.unpack("<d", f.read(8))[0])
                tl = struct.unpack("<Q", f.read(8))[0]
                f.seek(8 * tl, 1)
                ids.append(pid)
    pts = Points3D(np.array(xyz).reshape(-1, 3), np.array(rgb, dtype=np.uint8).reshape(-1, 3),
                   np.array(err), np.array(ids, dtype=np.uint64))
    return SparseModel(cams, imgs, pts)


def read_model(folder) -> SparseModel:
    folder = Path(folder)
    if (folder / "cameras.bin").exists():
        return _read_binary(folder)
    if (folder / "cameras.txt").exists():
        return _read_text(folder)
    raise FileNotFoundError(f"no COLMAP model (cameras.bin/.txt) in {folder}")


# ===================================================================== helpers
def write_points_ply(path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Plain coloured point-cloud PLY: the 'visual check' + three.js fallback viewer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rec = np.empty(n, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"] = xyz.astype(np.float32)
    rec["c"] = rgb.astype(np.uint8)
    with path.open("wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


def camera_centers(model: SparseModel) -> np.ndarray:
    out = []
    for im in sorted(model.images.values(), key=lambda i: i.name):
        out.append(-im.R.T @ im.tvec)
    return np.array(out).reshape(-1, 3)
