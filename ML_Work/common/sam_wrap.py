"""Thin wrappers around SAM 2.1 (Meta, Apache-2.0 code and weights) - the only GPU-model code of the Week 2 masks.

    pip install --no-deps git+https://github.com/facebookresearch/sam2.git      (scripts/setup_kaggle.sh sam2)
    weights come from the Hugging Face hub ("facebook/sam2.1-hiera-large", also -base-plus / -small / -tiny).

T4 note: no bf16 tensor cores, so autocast uses float16 by default (--sam-dtype float32 if you ever see NaN masks).
Everything is imported lazily so the rest of the pipeline (and the unit tests) never need sam2 installed.
The API calls are the documented ones (SAM2ImagePredictor.predict, SAM2VideoPredictor.add_new_mask/propagate_in_video).
Not run in this sandbox (no GPU): run `python tools/w2_selftest.py` once on Kaggle after setup.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "facebook/sam2.1-hiera-large"


def _torch():
    import torch
    return torch


def _autocast(dtype: str, device: str):
    torch = _torch()
    if device.startswith("cuda") and dtype in ("float16", "bfloat16"):
        return torch.autocast("cuda", dtype=getattr(torch, dtype))
    return contextlib.nullcontext()


def _need_sam2():
    try:
        import sam2  # noqa: F401
    except ImportError as e:
        raise RuntimeError("sam2 is not installed. Run:  bash scripts/setup_kaggle.sh sam2   "
                           "(pip install --no-deps git+https://github.com/facebookresearch/sam2.git)") from e


class SamImage:
    """Click / box / mask-refinement prompts on a single image."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cuda", dtype: str = "float16"):
        _need_sam2()
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.device, self.dtype = device, dtype
        self.pred = SAM2ImagePredictor.from_pretrained(model_id, device=device) if _accepts_device(
            SAM2ImagePredictor.from_pretrained) else SAM2ImagePredictor.from_pretrained(model_id)
        self._img_id = None

    def segment(self, rgb: np.ndarray, points=None, box=None, multimask: bool | None = None):
        """points: [(x, y, 1|0), ...] in image pixels. Returns (mask bool HxW, score float).
        SAM's advice: one ambiguous click -> ask for 3 masks and keep the best; several clicks -> a single mask."""
        torch = _torch()
        pts = np.array([[p[0], p[1]] for p in points], np.float32) if points else None
        lab = np.array([int(p[2]) for p in points], np.int32) if points else None
        if multimask is None:
            multimask = bool(points) and len(points) == 1 and box is None
        with torch.inference_mode(), _autocast(self.dtype, self.device):
            self.pred.set_image(rgb)
            masks, scores, _ = self.pred.predict(point_coords=pts, point_labels=lab,
                                                 box=None if box is None else np.asarray(box, np.float32),
                                                 multimask_output=multimask)
        best = int(np.argmax(scores))
        return masks[best] > 0, float(scores[best])


class SamVideo:
    """Propagate prompts through the frame folder (names '000001.jpg' ... are the integer stems SAM 2 sorts by)."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cuda", dtype: str = "float16"):
        _need_sam2()
        from sam2.build_sam import build_sam2_video_predictor_hf
        self.device, self.dtype = device, dtype
        self.pred = build_sam2_video_predictor_hf(model_id, device=device)

    def propagate(self, frames_dir: Path, anchors: dict[int, np.ndarray], start: int, progress=None) -> dict:
        """anchors: {frame_index: bool mask} (the click mask and any user-marked frames). Returns
        {frame_index: (bool mask, confidence)} for every frame, both directions from `start`."""
        torch = _torch()
        out: dict[int, tuple[np.ndarray, float]] = {}
        n_total = len([p for p in Path(frames_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg")])
        with torch.inference_mode(), _autocast(self.dtype, self.device):
            state = self.pred.init_state(video_path=str(frames_dir), offload_video_to_cpu=True)
            for fi, m in anchors.items():
                self.pred.add_new_mask(inference_state=state, frame_idx=int(fi), obj_id=1, mask=np.asarray(m, bool))
            for reverse in (False, True):
                for k, (fi, _ids, logits) in enumerate(self.pred.propagate_in_video(state, start_frame_idx=int(start),
                                                                                    reverse=reverse)):
                    lg = logits[0, 0].float().cpu().numpy()
                    m = lg > 0
                    conf = float((1 / (1 + np.exp(-lg[m]))).mean()) if m.any() else 0.0
                    if fi not in out:
                        out[fi] = (m, conf)
                    if progress:
                        progress(len(out) / max(n_total, 1))
            self.pred.reset_state(state)
        return out


def _accepts_device(fn) -> bool:
    import inspect
    try:
        return "device" in inspect.signature(fn).parameters or any(
            p.kind == p.VAR_KEYWORD for p in inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
