"""Central configuration + the on-disk workspace contract.

The directory layout below IS the "Week 2 handoff": Week 2 code reads these
exact paths. Do not rename things without updating docs/WEEK2_HANDOFF.md.

    <root>/<project_id>/
        video/input.<ext>
        frames/images/000001.jpg ...        resized + blur-filtered training frames
        frames/frames_index.json            per-frame metadata (source time, blur score)
        poses/vggt/                         COLMAP-style folder  (images/, sparse/0/)
            depth/000001.npy, conf/000001.npy   per-frame depth + confidence (Week 2 "lift to 3D")
            vggt_meta.json
        poses/glomap/                       COLMAP-style folder  (images/, sparse/0/)
        scene/                              the CHOSEN method (copy/symlink of poses/<method>)
        gs/                                 trainer output (ckpts, ply, stats)
        export/scene.ply, scene.splat, scene_meta.json
        metrics/<stage>.json                runtime + peak VRAM per stage
        manifest.json                       machine-readable index of everything above
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class FrameConfig:
    fps: str = "auto"            # "auto" or a number 1-4 (spec: 1-4 fps)
    min_fps: float = 1.0
    max_fps: float = 4.0
    long_side: int = 960         # spec: long side 640-960 px. NEVER feed 4K.
    max_frames: int = 400        # hard cap so VGGT/GLOMAP/3DGS stay inside a T4 session
    blur_rel: float = 0.5        # drop if Laplacian variance < blur_rel * median
    blur_abs: float = 15.0       # ... or below this absolute floor
    max_drop_fraction: float = 0.30  # never drop more than this share of frames
    jpg_quality: int = 95
    # auto-fps target: median inter-frame motion as a fraction of image width
    target_motion: float = 0.06


@dataclass
class VGGTConfig:
    chunk_size: int = 32         # frames per forward pass (memory grows with frames)
    overlap: int = 8             # frames shared between consecutive chunks
    dtype: str = "auto"          # "auto" -> float16 on T4 (cc 7.5), bfloat16 if cc >= 8
    conf_percentile: float = 40  # drop the lowest X% confidence pixels from the point cloud
    max_points: int = 300_000    # points3D size cap (sparse init for 3DGS)
    pixel_stride: int = 4        # subsample pixels before the cap
    model_id: str = "facebook/VGGT-1B"
    vggt_size: int = 518         # VGGT working width


@dataclass
class GlomapConfig:
    camera_model: str = "SIMPLE_RADIAL"   # undistorted afterwards -> PINHOLE for 3DGS
    single_camera: bool = True            # one phone = one camera
    sequential_overlap: int = 12          # neighbours matched per frame
    loop_detection: bool = False          # needs a vocab tree file; off by default
    use_gpu: bool = True
    max_num_features: int = 8192


@dataclass
class TrainConfig:
    trainer: str = "gsplat"      # "gsplat" | "inria"
    quick_iters: int = 7000      # spec: ~7k to confirm the pipeline works
    full_iters: int = 30000      # spec: 15-30k for quality
    iters: int = 0               # 0 -> use quick/full depending on --mode
    holdout_every: int = 8       # every Nth frame is held out for PSNR/SSIM
    data_factor: int = 1
    gsplat_dir: str = ""         # path to gsplat checkout containing examples/simple_trainer.py
    extra_args: list = field(default_factory=list)


@dataclass
class Config:
    root: str = "workspace"
    project_id: str = "demo"
    frames: FrameConfig = field(default_factory=FrameConfig)
    vggt: VGGTConfig = field(default_factory=VGGTConfig)
    glomap: GlomapConfig = field(default_factory=GlomapConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ---- paths -------------------------------------------------------
    @property
    def project(self) -> Path:
        return Path(self.root) / self.project_id

    def p(self, *parts: str) -> Path:
        return self.project.joinpath(*parts)

    @property
    def video_dir(self): return self.p("video")
    @property
    def frames_dir(self): return self.p("frames")
    @property
    def images_dir(self): return self.p("frames", "images")
    @property
    def frames_index(self): return self.p("frames", "frames_index.json")
    def poses_dir(self, method: str): return self.p("poses", method)
    @property
    def scene_dir(self): return self.p("scene")
    @property
    def gs_dir(self): return self.p("gs")
    @property
    def export_dir(self): return self.p("export")
    @property
    def metrics_dir(self): return self.p("metrics")
    @property
    def manifest_path(self): return self.p("manifest.json")

    def ensure(self) -> None:
        for d in (self.video_dir, self.images_dir, self.scene_dir, self.gs_dir,
                  self.export_dir, self.metrics_dir):
            d.mkdir(parents=True, exist_ok=True)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def add_common_args(ap) -> None:
    """Every stage script shares these flags so they can be run on their own."""
    ap.add_argument("--root", default=os.environ.get("W1_ROOT", "workspace"))
    ap.add_argument("--project", default=os.environ.get("W1_PROJECT", "demo"))
    ap.add_argument("--task-id", default=os.environ.get("W1_TASK_ID", ""),
                    help="backend task id; progress is POSTed to /tasks/{id}/progress")
    ap.add_argument("--backend-url", default=os.environ.get("W1_BACKEND_URL", ""),
                    help="e.g. https://xyz.trycloudflare.com ; empty = log locally only")


def config_from_args(args) -> Config:
    cfg = Config(root=args.root, project_id=args.project)
    cfg.ensure()
    return cfg


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)
