"""Image-editing models behind one interface: `edit(images, prompt, seed, size) -> PIL image`.

`images[0]` is the frame crop to edit, `images[1:]` are reference images (the chosen reference edit, optionally the
previous edited frame). Real models are loaded lazily, so the CPU tests never import torch.

Model code follows the official cards (read 2026-11-06):
- FLUX 3 Image Klein: `ModularPipeline.from_pretrained(...)`, `load_components(dtype={"vae": fp32, "default": bf16})`,
  `pipe(prompt=, image=[...], generator=, output="images")[0]`, upsampler removed via `blocks.sub_blocks.pop(...)`.
- Qwen-Image-2.1(-Turbo): `QwenImage21Pipeline.from_pretrained(..., torch_dtype=bf16)`,
  `pipe(prompt=, image=[...], width=, height=, generator=, [use_kv_cache=True | num_inference_steps=40]).images[0]`.
Anything marked [T4] below is not documented for these models and is measured by the benchmark on Kaggle.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# Precision options from the research doc §4. Weights storage / compute dtype.
PRECISIONS = {
    "bf16":      {"option": "A", "storage": "bfloat16", "compute": "bfloat16"},
    "bf16-fp32": {"option": "B", "storage": "bfloat16", "compute": "float32"},
    "fp8-bf16":  {"option": "D", "storage": "float8_e4m3fn", "compute": "bfloat16"},
    "fp16":      {"option": "C", "storage": "float16", "compute": "float16"},
}
PLACEMENTS = ("cuda", "split", "offload")

MODELS = {
    "klein": "black-forest-labs/flux-3-image-klein",
    "qwen-turbo": "Qwen/Qwen-Image-2.1-Turbo",
    "qwen": "Qwen/Qwen-Image-2.1",
    "fake": None,
}

# How the extra images are referred to in the prompt. Klein names inputs <ref_image_0>, <ref_image_1>, ... (its
# box docs); Qwen has no documented syntax, so plain words. To be tuned on real frames.
REF_TEMPLATES = {
    "klein": "{prompt}. Edit only <ref_image_0>. The edited object must look exactly like the edited object in "
             "<ref_image_1>: same colour, material and shape.",
    "qwen": "{prompt}. Edit only the first image. The edited object must look exactly like the edited object in "
            "the second image: same colour, material and shape.",
}


class PrecisionError(ValueError):
    pass


def check_precision(precision: str, allow_fp16: bool) -> dict:
    """The spec asks for bfloat16 (not float16). fp16 is never chosen silently: it needs an explicit flag."""
    if precision not in PRECISIONS:
        raise PrecisionError(f"unknown precision {precision!r}; choose one of {sorted(PRECISIONS)}")
    if precision == "fp16" and not allow_fp16:
        raise PrecisionError("fp16 compute is option C (not the spec's bf16): pass --allow-fp16 explicitly, "
                             "after the team decided it, and check the NaN/flat-image flags in the report")
    return PRECISIONS[precision]


def ref_prompt(model: str, prompt: str, n_images: int) -> str:
    if n_images <= 1:
        return prompt
    return REF_TEMPLATES["klein" if model == "klein" else "qwen"].format(prompt=prompt)


@dataclass
class CallStats:
    seconds: float
    peak_vram_mb: dict = field(default_factory=dict)  # "cuda:0" -> MB
    degenerate: bool = False  # NaN -> black, or a flat image: the fp16 overflow symptom


def degenerate_image(img: Image.Image) -> bool:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    return bool(a.std() < 1.0)


class Editor:
    model = "base"

    def info(self) -> dict:
        return {"model": self.model}

    def edit(self, images: list[Image.Image], prompt: str, seed: int, size: tuple[int, int]) -> tuple[Image.Image, CallStats]:
        raise NotImplementedError


class FakeEditor(Editor):
    """CPU stand-in for tests: tints the image with a colour from the prompt, pulls it towards the reference's
    colour when one is given, and deliberately changes *every* pixel a little (like a real model leaking outside the
    mask), so the composite guarantee is actually exercised."""
    model = "fake"

    def __init__(self, seconds: float = 0.0):
        self.seconds = seconds
        self.calls: list[dict] = []

    def edit(self, images, prompt, seed, size):
        t = time.perf_counter()
        h = hashlib.sha256(prompt.encode()).digest()
        img = np.asarray(images[0].convert("RGB").resize(size), dtype=np.float32)
        noise = np.random.default_rng(seed).normal(0, 6, img.shape)
        if len(images) > 1:  # copy the reference object's colour (centre of the reference crop = the object)
            ref = np.asarray(images[1].convert("RGB"), dtype=np.float32)
            rh, rw = ref.shape[:2]
            colour = np.median(ref[rh * 2 // 5: rh * 3 // 5, rw * 2 // 5: rw * 3 // 5].reshape(-1, 3), axis=0)
            out = np.clip(colour + noise, 0, 255).astype(np.uint8)
        else:
            colour = np.array([h[0], h[1], h[2]], dtype=np.float32)
            out = np.clip(0.3 * img + 0.7 * colour + noise, 0, 255).astype(np.uint8)
        if self.seconds:
            time.sleep(self.seconds)
        self.calls.append({"n_images": len(images), "prompt": prompt, "seed": seed, "size": size})
        return Image.fromarray(out), CallStats(seconds=time.perf_counter() - t)


class DiffusersEditor(Editor):
    """Shared torch plumbing: dtype policy, device placement, peak-VRAM measurement."""

    def __init__(self, model: str, precision: str, placement: str, allow_fp16: bool = False,
                 steps: int | None = None):
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}")
        self.model, self.precision, self.placement, self.steps = model, precision, placement, steps
        self.policy = check_precision(precision, allow_fp16)
        import torch  # noqa: F401  (Kaggle only)
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU: real models only run on the Kaggle worker (use --model fake on CPU)")
        t = time.perf_counter()
        self.pipe = self._load()
        self._apply_storage()
        self._place()
        self.load_seconds = round(time.perf_counter() - t, 1)

    # -- dtype -----------------------------------------------------------------------------------------------
    def _dt(self, name: str):
        return getattr(self.torch, name)

    def _load_dtype(self):
        """dtype the weights are loaded in. fp8 storage is applied after loading (layerwise casting)."""
        return self._dt("float16") if self.precision == "fp16" else self._dt("bfloat16")

    def _transformer(self):
        tr = getattr(self.pipe, "transformer", None)
        if tr is None:
            raise RuntimeError(f"{self.model}: pipeline has no .transformer; precision {self.precision} needs it")
        return tr

    def _apply_storage(self):
        """Options B and D: keep weights in `storage`, up-cast each layer to `compute` just before it runs.
        [T4] Whether diffusers' layerwise casting works with these two (new) transformers is measured, not known."""
        storage, compute = self.policy["storage"], self.policy["compute"]
        if self.precision in ("bf16-fp32", "fp8-bf16"):
            self._transformer().enable_layerwise_casting(storage_dtype=self._dt(storage), compute_dtype=self._dt(compute))

    # -- placement -------------------------------------------------------------------------------------------
    def _modules(self) -> dict:
        comps = getattr(self.pipe, "components", None)
        items = comps.items() if isinstance(comps, dict) else vars(self.pipe).items()
        return {k: v for k, v in items if isinstance(v, self.torch.nn.Module)}

    def _place(self):
        if self.placement == "offload":
            if not hasattr(self.pipe, "enable_model_cpu_offload"):
                raise RuntimeError(f"{self.model}: pipeline has no enable_model_cpu_offload")
            self.pipe.enable_model_cpu_offload()
        elif self.placement == "cuda":
            self.pipe.to("cuda:0")
        else:  # split: text encoder(s) on GPU 1, everything else on GPU 0 (research §5). [T4] to verify.
            if self.torch.cuda.device_count() < 2:
                raise RuntimeError("placement 'split' needs 2 GPUs")
            for name, mod in self._modules().items():
                mod.to("cuda:1" if name.startswith("text_encoder") else "cuda:0")

    # -- calls -----------------------------------------------------------------------------------------------
    def info(self) -> dict:
        t = self.torch
        return {"model": self.model, "repo": MODELS[self.model], "precision": self.precision,
                "option": self.policy["option"], "storage_dtype": self.policy["storage"],
                "compute_dtype": self.policy["compute"], "placement": self.placement,
                "load_seconds": self.load_seconds, "steps": self.steps or "model default",
                "gpus": [t.cuda.get_device_name(i) for i in range(t.cuda.device_count())],
                "torch": t.__version__}

    def edit(self, images, prompt, seed, size):
        t = self.torch
        for i in range(t.cuda.device_count()):
            t.cuda.reset_peak_memory_stats(i)
        start = time.perf_counter()
        img = self._call(images, ref_prompt(self.model, prompt, len(images)), seed, size)
        t.cuda.synchronize()
        stats = CallStats(seconds=round(time.perf_counter() - start, 2),
                          peak_vram_mb={f"cuda:{i}": round(t.cuda.max_memory_allocated(i) / 2**20)
                                        for i in range(t.cuda.device_count())},
                          degenerate=degenerate_image(img))
        return img, stats

    def _load(self):
        raise NotImplementedError

    def _call(self, images, prompt, seed, size):
        raise NotImplementedError


class KleinEditor(DiffusersEditor):
    def __init__(self, *a, upsampler: bool = False, **kw):
        self.upsampler = upsampler
        super().__init__("klein", *a, **kw)

    def _load(self):
        from diffusers import ModularPipeline
        repo = MODELS["klein"]
        if self.upsampler:
            pipe = ModularPipeline.from_pretrained(repo)
        else:  # the 9B upsampler does not fit next to the model on a T4 (research §5); prompts are written by us
            blocks = ModularPipeline.from_pretrained(repo).blocks
            blocks.sub_blocks.pop("prompt_upsampler")
            pipe = blocks.init_pipeline(repo)
        pipe.load_components(dtype={"vae": self._dt("float32"), "default": self._load_dtype()})
        return pipe

    def _call(self, images, prompt, seed, size):
        # The card documents no height/width: the output follows the input image, which is already `size`.
        kw = {"num_inference_steps": self.steps} if self.steps else {}
        return self.pipe(prompt=prompt, image=images if len(images) > 1 else images[0],
                         generator=self.torch.Generator().manual_seed(seed), output="images", **kw)[0]

    def info(self):
        return {**super().info(), "prompt_upsampler": self.upsampler}


class QwenEditor(DiffusersEditor):
    def __init__(self, model: str, *a, **kw):
        super().__init__(model, *a, **kw)

    def _load(self):
        from diffusers import QwenImage21Pipeline
        return QwenImage21Pipeline.from_pretrained(MODELS[self.model], torch_dtype=self._load_dtype())

    def _call(self, images, prompt, seed, size):
        kw = {"use_kv_cache": True} if self.model == "qwen-turbo" else {"num_inference_steps": self.steps or 40}
        return self.pipe(prompt=prompt, image=images if len(images) > 1 else images[0], width=size[0],
                         height=size[1], generator=self.torch.Generator("cpu").manual_seed(seed), **kw).images[0]


def make_editor(model: str, precision: str = "bf16", placement: str = "split", allow_fp16: bool = False,
                steps: int | None = None, upsampler: bool = False) -> Editor:
    check_precision(precision, allow_fp16)  # before any download
    if model == "fake":
        return FakeEditor()
    if model == "klein":
        return KleinEditor(precision, placement, allow_fp16=allow_fp16, steps=steps, upsampler=upsampler)
    if model in ("qwen", "qwen-turbo"):
        return QwenEditor(model, precision, placement, allow_fp16=allow_fp16, steps=steps)
    raise ValueError(f"unknown model {model!r}; choose one of {sorted(MODELS)}")
