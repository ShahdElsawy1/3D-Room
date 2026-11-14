"""Diffusion editing of the tracked object (Week 2, brief section 4.3).

Everything here that does not need a GPU is plain numpy/cv2 so it is unit-tested on CPU:
    fit_to_multiple / unfit     frames are 640-960 px but not multiples of 16: pad (or resize), edit, map back
    outline_hint                paint a ring just OUTSIDE the mask so the model knows which object is meant
    composite                   result = original everywhere except inside the (slightly grown) mask
    pick_reference              the frame where the object is large and fully visible
    build_prompt                the edit prompt, with the wording that refers to the reference edit
    validate_edited             contract check before the rebuild reads edits/E/edited/

The only GPU code is `QwenEditor` (Qwen-Image-2.1 through diffusers). Its call follows the diffusers model page
(QwenImage21Pipeline.__call__): `image=` takes one image or a list (the order is the order the model reads them), and
there is NO mask argument. The mask is therefore used in two ways: as an outline hint on the input image, and as the
compositing region that guarantees nothing outside the mask changes.

Not run in the sandbox that wrote it (no GPU): run `w2_edit.py --limit 2 --n-seeds 1` once on Kaggle as the smoke test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# fragmentation was part of the T4 OOM in the first Week 2 run; must be set before torch touches CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np

MULTIPLE = 16


# ----------------------------------------------------------------------------- sizes
def fit_to_multiple(img: np.ndarray, multiple: int = MULTIPLE, mode: str = "pad") -> tuple[np.ndarray, dict]:
    """Image (HxW or HxWxC) -> image whose sides are multiples of `multiple`, plus what `unfit` needs to undo it.
    pad    : reflect-pad right/bottom (keeps every pixel where it was, so mask compositing stays exact)
    resize : resample to the next multiple (changes pixel positions slightly)"""
    h, w = img.shape[:2]
    H, W = -(-h // multiple) * multiple, -(-w // multiple) * multiple
    info = {"mode": mode, "orig_hw": (h, w), "size_hw": (H, W)}
    if (H, W) == (h, w):
        return img, info
    if mode == "pad":
        return cv2.copyMakeBorder(img, 0, H - h, 0, W - w, cv2.BORDER_REFLECT_101), info
    if mode == "resize":
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC), info
    raise ValueError(f"unknown mode {mode!r}")


def unfit(img: np.ndarray, info: dict) -> np.ndarray:
    """Back to the original frame size. The model may return a slightly different size than asked: resample to the
    padded size first, then crop (pad) or resample (resize)."""
    h, w = info["orig_hw"]
    H, W = info["size_hw"]
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
    if info["mode"] == "pad":
        return img[:h, :w]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if (h, w) != (H, W) else img


def gen_size(size_hw: tuple[int, int], scale: float = 1.0, multiple: int = MULTIPLE) -> tuple[int, int]:
    """Size to ask the model for: the padded frame times `scale`, rounded to a multiple."""
    return tuple(max(multiple, int(round(v * scale / multiple)) * multiple) for v in size_hw)  # type: ignore[return-value]


# ----------------------------------------------------------------------------- mask use
def _k(px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))


def grow_mask(mask: np.ndarray, px: int) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    return (cv2.dilate(m, _k(px)) > 0) if px > 0 else m > 0


def outline_hint(rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), thickness: int = 3, gap: int = 2) -> np.ndarray:
    """Copy of `rgb` with a ring painted `gap` px outside the mask. Nothing inside the mask is touched, and the final
    image takes its pixels outside the mask from the ORIGINAL frame, so the ring never reaches the output."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    inner = cv2.dilate(m, _k(gap)) > 0
    outer = cv2.dilate(m, _k(gap + thickness)) > 0
    out = rgb.copy()
    out[outer & ~inner] = color
    return out


def composite(orig: np.ndarray, edited: np.ndarray, mask: np.ndarray, grow: int = 2, feather: int = 2) -> np.ndarray:
    """orig/edited: HxWx3 uint8 (same size). Returns an image that equals `orig` bit for bit everywhere outside the mask
    grown by `grow` px; inside it the edit is blended in with a soft edge of `feather` px (the edge pixels of the grown
    region only, the mask interior is the pure edit)."""
    if orig.shape != edited.shape:
        raise ValueError(f"orig {orig.shape} != edited {edited.shape}")
    core = np.asarray(mask) > 0
    region = grow_mask(core, grow)
    alpha = region.astype(np.float32)
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(feather / 2.0, 0.5)) * region
        alpha[cv2.erode(core.astype(np.uint8), _k(feather)) > 0] = 1.0
    a = alpha[..., None]
    out = np.clip(orig.astype(np.float32) * (1 - a) + edited.astype(np.float32) * a + 0.5, 0, 255).astype(np.uint8)
    out[~region] = orig[~region]                       # exact, not "almost": float rounding must not leak outside
    return out


def outside_diff(orig: np.ndarray, result: np.ndarray, mask: np.ndarray, grow: int = 0) -> dict:
    """How much `result` differs from `orig` OUTSIDE the mask grown by `grow` px (0 = the mask itself)."""
    outside = ~grow_mask(mask, grow)
    d = np.abs(result.astype(np.int16) - orig.astype(np.int16)).max(axis=2)[outside]
    return {"max_abs_diff": int(d.max()) if d.size else 0, "share_changed": float((d > 0).mean()) if d.size else 0.0}


def pick_reference(candidates: dict[str, np.ndarray]) -> str:
    """Frame where the object is large and clearly visible: biggest mask area, halved when the mask touches the border
    (the object is cut off by the frame edge, so the model would invent the missing part)."""
    if not candidates:
        raise ValueError("no frame contains the object")

    def score(m: np.ndarray) -> float:
        m = np.asarray(m) > 0
        touches = m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
        return float(m.sum()) * (0.5 if touches else 1.0)

    return max(sorted(candidates), key=lambda n: score(candidates[n]))


def build_prompt(user_prompt: str, with_reference: bool, hint: bool) -> str:
    base = user_prompt.strip().rstrip(".")
    parts = []
    if with_reference:
        parts += ["Image 1 is the photo to edit. Image 2 is the same room from another viewpoint with the finished edit.",
                  f"{base}.",
                  "Give the object in image 1 exactly the same material, colour and style as the edited object in image 2."]
    else:
        parts.append(f"{base}.")
    if hint:
        parts.append("Only change the object enclosed by the red outline. Do not draw the outline. "
                     "Keep everything else exactly as it is.")
    return " ".join(parts)


def contact_sheet(pairs: list[tuple[np.ndarray, np.ndarray]], cols: int = 3, thumb_w: int = 320) -> np.ndarray:
    """[(original, edited)] -> grid with each original above its edit, for a quick look at colour drift between views."""
    tiles = []
    for o, e in pairs:
        h = int(o.shape[0] * thumb_w / o.shape[1])
        tiles.append(np.vstack([cv2.resize(o, (thumb_w, h), interpolation=cv2.INTER_AREA),
                                cv2.resize(e, (thumb_w, h), interpolation=cv2.INTER_AREA)]))
    if not tiles:
        return np.zeros((8, 8, 3), np.uint8)
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def validate_edited(project: Path, edit_id: str) -> list[str]:
    """Contract check for edits/E/edited (what w2_rebuild reads). Empty list = OK."""
    E = Path(project) / "edits" / edit_id
    ip = E / "edited" / "index.json"
    if not ip.exists():
        return [f"{ip} missing"]
    idx = json.loads(ip.read_text())
    problems, seen = [], set()
    for e in idx.get("edited", []):
        name, f = e.get("name"), e.get("file")
        p = E / str(f)
        fr = Path(project) / "frames" / "images" / str(name)
        if name in seen:
            problems.append(f"{name}: listed twice")
        seen.add(name)
        if not fr.exists():
            problems.append(f"{name}: not a frame of frames/images")
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            problems.append(f"{name}: {f} missing")
        elif img.shape[:2] != cv2.imread(str(fr), cv2.IMREAD_COLOR).shape[:2]:
            problems.append(f"{name}: size {img.shape[1::-1]} != frame")
    return problems


# ----------------------------------------------------------------------------- the model (GPU)
class QwenEditor:
    """Qwen-Image-2.1 (7B DiT, Qwen3-VL text/vision encoder) through diffusers' QwenImage21Pipeline.

    quant   none | 8bit | 4bit   bitsandbytes on the transformer AND the text encoder (the two big parts); the VAE stays dense
    offload none | model | sequential   accelerate CPU offload, for when everything does not fit on one T4
    dtype   bfloat16 (the brief says bf16, not fp16) | float16 | float32
    GPU 0 is meant to hold the whole thing (brief: "GPU 0 holds the diffusion model, text encoder and VAE");
    pick it with the stage's --gpu (CUDA_VISIBLE_DEVICES).
    """

    def __init__(self, model_id: str = "Qwen/Qwen-Image-2.1", dtype: str = "bfloat16", quant: str = "4bit",
                 offload: str = "none", steps: int = 40, cfg: float = 1.0, negative_prompt: str = "",
                 vae_tiling: bool = True):
        try:
            import torch
            from diffusers import QwenImage21Pipeline
        except ImportError as e:
            raise RuntimeError("torch/diffusers with QwenImage21Pipeline is not available. Run:  "
                               "bash scripts/setup_kaggle.sh diffusion   (diffusers + transformers>=5.17 + bitsandbytes)") from e
        self.torch, self.steps, self.cfg, self.negative = torch, steps, cfg, negative_prompt
        td = getattr(torch, dtype)
        kw: dict = {"torch_dtype": td}
        if quant != "none":
            from diffusers.quantizers import PipelineQuantizationConfig
            if quant == "4bit":
                qc = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": td},
                    components_to_quantize=["transformer", "text_encoder"])
            elif quant == "8bit":
                qc = PipelineQuantizationConfig(quant_backend="bitsandbytes_8bit", quant_kwargs={"load_in_8bit": True},
                                                components_to_quantize=["transformer", "text_encoder"])
            else:
                raise ValueError(f"unknown quant {quant!r}")
            kw["quantization_config"] = qc
        if offload == "none":
            kw["device_map"] = "cuda"
        self.pipe = QwenImage21Pipeline.from_pretrained(model_id, **kw)
        if offload == "model":
            self.pipe.enable_model_cpu_offload()
        elif offload == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        if vae_tiling:
            try:
                self.pipe.vae.enable_tiling()
            except Exception:                                  # noqa: BLE001 - not every VAE has it; only a memory aid
                pass
        vdt = next(self.pipe.vae.parameters()).dtype
        if vdt == torch.float32 and dtype != "float32":
            print(f"[diffusion] WARNING: VAE is float32 although {dtype} was requested", flush=True)
        self.load_peak_mb = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()               # generation peak is measured separately from loading

    def peak_mb(self) -> float | None:
        t = self.torch
        return round(t.cuda.max_memory_allocated() / 2 ** 20, 1) if t.cuda.is_available() else None

    def edit(self, images: list[np.ndarray], prompt: str, seed: int, size_hw: tuple[int, int]) -> np.ndarray:
        """images: RGB uint8, already padded to multiples of 16; first = the frame to edit, then the reference edit.
        Returns RGB uint8 of about `size_hw` (the caller maps it back with `unfit`)."""
        from PIL import Image
        torch = self.torch
        h, w = size_hw
        # The Qwen3-VL vision tower sees every input image: at 960x544 its attention alone asked for >1 GiB on a T4
        # (the first-run OOM, with the weights already in 4 bit). Feed it images no larger than what we generate.
        pil = []
        for i in images:
            im = Image.fromarray(i)
            if im.size != (w, h):
                im = im.resize((w, h), Image.LANCZOS)
            pil.append(im)
        kw = dict(prompt=prompt, image=pil, height=h, width=w,
                  num_inference_steps=self.steps, generator=torch.Generator("cpu").manual_seed(int(seed)))
        if self.cfg > 1.0:
            kw.update(true_cfg_scale=self.cfg, negative_prompt=self.negative or " ")
        try:
            with torch.inference_mode():
                out = self.pipe(**kw).images[0]
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()                           # one retry: fragmentation often clears
            with torch.inference_mode():
                out = self.pipe(**kw).images[0]
        arr = np.asarray(out.convert("RGB"))
        if float(arr.std()) < 1.0:
            raise RuntimeError("the model returned a constant image - usually fp16 overflow or NaNs. Use --dtype bfloat16 "
                               "(or float32 with --offload model).")
        return arr
