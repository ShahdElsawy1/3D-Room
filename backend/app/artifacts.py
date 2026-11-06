"""Artifact catalogue, validation, safe archive extraction and atomic promotion.

Every artifact kind maps to ONE fixed location in Person A's layout (ML_Work/common/config.py), so names and
folders are exactly what the pipeline wrote and what Week 2 reads:

    kind         path                      type     contents
    frames       frames/                   folder   images/000001.jpg ..., frames_index.json
    poses        poses/                    folder   vggt/ and/or glomap/: sparse/0, depth, conf, points.ply, meta
    scene        scene/                    folder   the chosen poses: sparse/0 (poses + intrinsics + points3D),
                                                    images/, points.ply, chosen.json      <- 3DGS / Week 2 input
    gs           gs/                       folder   trainer output: ckpts/*.pt (Week 2 fine-tune), ply/, stats/, latest.json
    metrics      metrics/                  folder   per-stage runtime/VRAM json, logs, comparison report
    manifest     manifest.json             file
    splat        export/scene.splat        file     what the web viewer loads
    ply          export/scene.ply          file
    cameras      export/cameras.json       file
    scene_meta   export/scene_meta.json    file

Uploads of a task are staged in <data>/staging/<task_id>/ (same relative layout) and only promoted into the
project folder when the task completes. A failed or cancelled run therefore never leaves the project with,
for example, new frames next to an old splat.
"""
from __future__ import annotations

import json
import posixpath
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .models import Artifact, UploadSession
from .storage import project_dir


@dataclass(frozen=True)
class Kind:
    path: str
    is_dir: bool


KINDS: dict[str, Kind] = {
    "frames": Kind("frames", True),
    "poses": Kind("poses", True),
    "scene": Kind("scene", True),
    "gs": Kind("gs", True),
    "metrics": Kind("metrics", True),
    "manifest": Kind("manifest.json", False),
    "splat": Kind("export/scene.splat", False),
    "ply": Kind("export/scene.ply", False),
    "cameras": Kind("export/cameras.json", False),
    "scene_meta": Kind("export/scene_meta.json", False),
}
# Spec: frames, poses + intrinsics + sparse point cloud (scene/sparse/0) and the splat must be stored for Week 2;
# cameras.json + scene_meta.json are what the viewer needs.
REQUIRED = {"reconstruct": ["frames", "scene", "splat", "cameras", "scene_meta"]}

SPLAT_RECORD_BYTES = 32       # antimatter15 .splat: 3xf32 pos, 3xf32 scale, 4xu8 rgba, 4xu8 rot


class ArtifactError(ValueError):
    pass


def staging_dir(settings: Settings, task_id: str) -> Path:
    return settings.data_dir / "staging" / task_id


def parts_dir(settings: Settings) -> Path:
    d = settings.data_dir / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ safe extraction
def _clean_member_name(name: str) -> str:
    n = name.replace("\\", "/")
    while n.startswith("./"):
        n = n[2:]
    n = n.rstrip("/")
    if not n or n.startswith("/") or (len(n) > 1 and n[1] == ":"):
        raise ArtifactError(f"absolute path in archive: {name!r}")
    norm = posixpath.normpath(n)
    if norm == ".." or norm.startswith("../") or "/../" in f"/{n}/":
        raise ArtifactError(f"path traversal in archive: {name!r}")
    return norm


def extract_archive(archive: Path, root: str, dest: Path, search_roots: list[Path],
                    max_bytes: int, max_members: int = 200_000) -> dict:
    """Extract `archive` (tar / tar.gz) whose members must all live under `root` (e.g. "frames") into `dest`
    (a project-shaped folder). Regular files and folders only are written directly; symlinks/hardlinks are
    NOT created (unsafe, and unsupported on Windows) but materialised as copies of their target, which must
    resolve inside the project, in this archive or in `search_roots` (earlier uploads of the same task, then
    the project itself). Devices, FIFOs etc. are rejected."""
    files, total, links = 0, 0, []
    with tarfile.open(archive, "r:*") as tf:
        members = tf.getmembers()
        if len(members) > max_members:
            raise ArtifactError(f"archive has {len(members)} entries (limit {max_members})")
        for m in members:
            name = _clean_member_name(m.name)
            if name != root and not name.startswith(root + "/"):
                raise ArtifactError(f"archive entry {m.name!r} is outside '{root}/'")
            target = dest / name
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif m.isreg():
                total += m.size
                if total > max_bytes:
                    raise ArtifactError(f"archive expands to more than {max_bytes // 2**20} MB")
                target.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                files += 1
            elif m.issym() or m.islnk():
                link = m.linkname.replace("\\", "/")
                if link.startswith("/"):
                    raise ArtifactError(f"absolute link target in archive: {m.name!r} -> {m.linkname!r}")
                tgt = posixpath.normpath(posixpath.join(posixpath.dirname(name), link) if m.issym() else link)
                if tgt == ".." or tgt.startswith("../"):
                    raise ArtifactError(f"link escapes the project: {m.name!r} -> {m.linkname!r}")
                links.append((name, tgt))
            else:
                raise ArtifactError(f"unsupported entry type in archive: {m.name!r}")

    materialised, dangling = [], []
    for _ in range(5):                                       # link chains (scene/images -> poses/x/images -> ...)
        pending = []
        for name, tgt in links:
            src = next((r / tgt for r in [dest, *search_roots] if (r / tgt).exists()), None)
            if src is None:
                pending.append((name, tgt))
                continue
            out = dest / name
            out.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, out, dirs_exist_ok=True)
            else:
                shutil.copy2(src, out)
            materialised.append(f"{name} -> {tgt}")
        if len(pending) == len(links):
            break
        links = pending
    dangling = [f"{n} -> {t}" for n, t in links if not (dest / n).exists()]
    return {"files": files, "bytes": total, "links_materialised": materialised, "dangling_links": dangling}


# ------------------------------------------------------------------ per-kind validation
def _has(folder: Path, stem: str) -> bool:
    return any((folder / f"{stem}{ext}").is_file() for ext in (".bin", ".txt"))


def validate(kind: str, root: Path) -> dict:
    """`root` is the project-shaped folder containing the artifact. Raises ArtifactError with a clear reason."""
    k = KINDS[kind]
    p = root / k.path
    info: dict = {}
    if kind == "frames":
        imgs = sorted((p / "images").glob("*.jpg")) + sorted((p / "images").glob("*.png"))
        if not imgs:
            raise ArtifactError("frames/images/ contains no .jpg/.png frames")
        if not (p / "frames_index.json").is_file():
            raise ArtifactError("frames/frames_index.json is missing")
        _json(p / "frames_index.json")
        info["num_frames"] = len(imgs)
    elif kind == "scene":
        sp = p / "sparse" / "0"
        missing = [s for s in ("cameras", "images", "points3D") if not _has(sp, s)]
        if missing:
            raise ArtifactError(f"scene/sparse/0 is missing {', '.join(missing)} (.bin or .txt)")
        imgs = list((p / "images").glob("*")) if (p / "images").is_dir() else []
        if not imgs:
            raise ArtifactError("scene/images is empty or missing (a symlink whose target was not uploaded "
                                "first? upload frames and poses before scene)")
        info["num_images"] = len(imgs)
        info["has_points_ply"] = (p / "points.ply").is_file()
        if (p / "chosen.json").is_file():
            info["pose_method"] = _json(p / "chosen.json").get("method")
    elif kind == "poses":
        methods = [d.name for d in p.iterdir() if d.is_dir()] if p.is_dir() else []
        if not methods:
            raise ArtifactError("poses/ contains no method folder")
        info["methods"] = sorted(methods)
    elif kind in ("gs", "metrics"):
        if not p.is_dir() or not any(p.rglob("*")):
            raise ArtifactError(f"{k.path}/ is empty")
        if kind == "gs":
            info["checkpoints"] = sorted(str(x.relative_to(p).as_posix()) for x in p.rglob("*.pt"))
    elif kind == "splat":
        size = p.stat().st_size
        if size == 0 or size % SPLAT_RECORD_BYTES:
            raise ArtifactError(f"scene.splat size {size} is not a positive multiple of {SPLAT_RECORD_BYTES} bytes")
        info["num_gaussians"] = size // SPLAT_RECORD_BYTES
    elif kind == "ply":
        with p.open("rb") as f:
            if f.read(3) != b"ply":
                raise ArtifactError("scene.ply does not start with a PLY header")
    elif kind in ("cameras", "scene_meta", "manifest"):
        data = _json(p)
        if kind == "cameras":
            if not isinstance(data, list) or not data:
                raise ArtifactError("cameras.json must be a non-empty list")
            need = {"img_name", "width", "height", "position", "rotation", "fx", "fy", "cx", "cy"}
            if need - set(data[0]):
                raise ArtifactError(f"cameras.json entries miss {sorted(need - set(data[0]))}")
            info["num_cameras"] = len(data)
    return info


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ArtifactError(f"{path.name} is not valid JSON: {e}") from e


def tree_size(p: Path) -> int:
    return p.stat().st_size if p.is_file() else sum(x.stat().st_size for x in p.rglob("*") if x.is_file())


# ------------------------------------------------------------------ promotion (task complete)
def staged_sessions(db: Session, task_id: str) -> dict[str, UploadSession]:
    rows = db.scalars(select(UploadSession).where(UploadSession.task_id == task_id,
                                                  UploadSession.status == "complete")
                      .order_by(UploadSession.updated_at)).all()
    return {s.kind: s for s in rows}          # the latest upload of each kind wins


def missing_required(db: Session, task_type: str, task_id: str) -> list[str]:
    have = staged_sessions(db, task_id)
    return [k for k in REQUIRED.get(task_type, []) if k not in have]


def promote(db: Session, settings: Settings, task) -> list[Artifact]:
    """Move every staged artifact of `task` into the project folder (replacing the previous version of the same
    kind) and upsert its Artifact row. Caller commits."""
    stage = staging_dir(settings, task.id)
    pdir = project_dir(settings, task.project_id)
    out = []
    for kind, s in staged_sessions(db, task.id).items():
        src, dst = stage / s.path, pdir / s.path
        if not src.exists():
            raise ArtifactError(f"staged {kind} is missing on disk ({s.path})")
        dst.parent.mkdir(parents=True, exist_ok=True)
        old = dst.with_name(dst.name + ".old")
        if old.exists():
            shutil.rmtree(old) if old.is_dir() else old.unlink()
        if dst.exists():
            dst.rename(old)
        shutil.move(str(src), str(dst))
        if old.exists():
            shutil.rmtree(old) if old.is_dir() else old.unlink()

        a = db.scalar(select(Artifact).where(Artifact.project_id == task.project_id, Artifact.version == s.version,
                                             Artifact.kind == kind))
        if a is None:
            a = Artifact(project_id=task.project_id, kind=kind, version=s.version, path=s.path)
            db.add(a)
        a.task_id, a.path, a.sha256 = task.id, s.path, s.sha256
        a.size_bytes = tree_size(dst)
        a.meta = {**(s.meta or {}), "is_dir": s.is_archive, "upload_bytes": s.size_bytes}
        out.append(a)
    shutil.rmtree(stage, ignore_errors=True)
    return out
