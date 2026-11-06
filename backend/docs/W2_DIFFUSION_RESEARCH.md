# Week 2: diffusion editing model research (Person B, M6)

**Status: research only, for team review. Nothing is implemented yet.** Facts are from the official pages
listed at the end, read on 2026-11-06. Both models were released in Sept/Oct 2026 and have no independent
benchmarks yet. Anything marked **[estimate]** or **[to measure]** is reasoning, not a published number, and has
to be confirmed on a Kaggle T4.

## 1. TL;DR

| | FLUX 3 Image Klein (distilled) | Qwen-Image-2.1 (+ Turbo) |
|---|---|---|
| Released | 2026-10-15 | 2026-09-20 (Turbo 2026-10-09) |
| Image transformer | **5.6B** | **7B** (32 single-stream DiT layers) |
| Text encoder | Qwen3-VL-4B-Instruct | Qwen3-VL **8B**, which also encodes the condition images |
| Extra model | prompt upsampler, 9B (18.8 GB). "Always run", but can be removed | optional prompt rewriters, 9B |
| Steps / CFG | 16 / none (base: 50 / 3.5) | 40 (Turbo: **8**, CFG 1) |
| Instruction editing | yes | yes |
| **Binary mask input** | **no** (bounding boxes in the prompt only). Card: *"Boxes … do not constrain pixels to a hard mask"*, *"Edits can change content outside the requested region."* | intro mentions *"circles, painted annotations, or separate masks"*, but **no mask argument in any diffusers example** |
| Reference images | up to 10, `image=[...]` | up to 10, `image=[...]` |
| Diffusers | `ModularPipeline` (diffusers from git) | `QwenImage21Pipeline` (diffusers from git, transformers ≥ 5.17) |
| Official dtypes | BF16; FP8 (`variants/fp8r`); NVFP4 | BF16; ComfyUI `int8_convrot`, `w4a8` text encoder |
| Published memory | peak **31 GB BF16 / 26 GB FP8 / 25 GB NVFP4** (B200, 1024², upsampler off); "down to an RTX 3090" (24 GB) | **not stated**; CPU offload offered |
| T4 / Turing / fp16 | not mentioned | not mentioned |
| License | FLUX Kommunity License v1.0: non-commercial OK (+ commercial output for < US$5M revenue); **gated** on HF | Qwen Research License: **research/evaluation only**, non-commercial |
| bf16 weights on 2×T4 without quantization | **plausible** (11.2 GB + 8.8 GB on separate GPUs) **[estimate]** | **no**: 14 GB + ~16 GB, each near or over a 15 GB T4 |

**Recommendation (for review):** use **FLUX 3 Image Klein (distilled)** as the primary model and **Qwen-Image-2.1-Turbo**
as the comparison. Neither gives us mask-exact editing, so the mask guarantee comes from our own pipeline:
crop around the object, blend latents with the mask during sampling, then do a hard pixel composite. The open question
is **compute speed in bf16 on a T4**, which must be measured before M7 is designed in detail (§4, §6).

## 2. What each model gives us for the spec's procedure

Spec §4.3: first frame = frame + mask + prompt (several seeds, user picks). Other frames = frame + own mask + same prompt
+ **reference edit** as an extra conditioning image.

- **Reference conditioning is supported by both**: a list of up to 10 images. Our call would be `image=[frame_t_crop,
  reference_edit_crop]` (optionally `+ previous_edit_crop`, one of the spec's ideas). Neither card says how a
  reference "binds" to the edited object. Klein: *"reference images anchor content, they do not lock structure."*
  Consistency across views is therefore **[to measure]**, and it is exactly one of the problems the spec asks us to document.
- **Masks are not a model input**, so "conditioned on the mask" has to be built by us:
  1. **Crop to the mask's bounding box** (plus a margin) and edit only the crop. That means fewer tokens (faster,
     less VRAM), and the model mostly sees the object.
  2. **Latent blending during sampling** (blended-latent / RePaint style). After each denoising step, set the latent
     outside the (downsampled, dilated) mask to the original latent re-noised to the current level. With flow
     matching: `x_t = (1-σ_t)·x0 + σ_t·ε`. Diffusers' `callback_on_step_end` can modify latents. This keeps the
     background fixed while the object is generated in context. **[to verify]** whether Klein's ModularPipeline and
     Qwen's pipeline expose the output latents in a callback. If not, a thin custom sampling loop is needed.
  3. **Klein only:** also pass the mask's bounding box in the prompt (official feature: a JSON list of boxes on a
     0–1000 grid appended to the prompt).
  4. **Pixel composite (the guarantee, spec §1):** `out = orig·(1−m) + edit·m`, where `m` is 0 everywhere outside the
     dilated mask. Every pixel outside it is **bit-identical** to the original, by construction and checked by a test.
     Feathering only happens in a band inside the dilated mask.
- **Several seeds for the first frame:** both pipelines take a `generator`. This is cheap to support.
- **Same seed and prompt for every frame** (spec idea): trivial. **2×2 tiling** (spec idea): possible because both work at
  1–2K, but a 2×2 grid of crops multiplies tokens by 4 **[to measure]**.

## 3. Resolution

- Frames are 960×540 (Week 1). **540 is not divisible by 16 (or 32)**. Spec §3: resize or pad, then map back.
- The models are trained around 1–2K (Klein 768sq–4K; Qwen native 2K, presets all multiples of 16). Neither page states
  a divisibility rule. **Assumption to verify:** use multiples of 32 (16× autoencoder × 2 patch is common), or 16 if a test
  shows it is enough.
- Plan: crop box (margin ~15 %) → resize so the long side is in {512, 768, 1024} (a setting to measure) and both sides
  are multiples of 32 → edit → resize back to the crop's original size → composite into the 960×540 frame. The stored
  frame stays at Week 1 resolution. Frame size and names don't change, so the COLMAP model stays valid for the rebuild.

## 4. The bf16 question on a T4 (the critical constraint)

**Facts**
- The T4 is Turing, **compute capability 7.5**. Its tensor cores do fp16 (and int8/int4), **not bf16**. Native bf16 needs
  compute capability ≥ 8.0 (Ampere). PyTorch still *runs* many bf16 ops on a T4 through slower non-tensor-core or
  emulated paths. One reported fine-tuning benchmark was ~4× slower than fp16. Some kernels fail outright (reported:
  cuDNN bf16 conv2d crash on a T4), and `torch.cuda.is_bf16_supported()` can return True on a T4, so it can't be trusted.
- Both models are **published in bf16 only**, and every official example uses `torch.bfloat16` (Klein: VAE in float32).
  Neither page mentions fp16. That is consistent with the spec's "BFloat16 (not float16)": these models are trained in
  bf16, and fp16's smaller range risks overflow (NaN or black images) in activations.
- **fp8 does not solve compute on a T4.** fp8 tensor-core math needs compute capability ≥ 8.9 (Ada/Hopper). On a T4, fp8 can
  only be a **storage** format: weights are kept in fp8 and up-cast per layer before each matmul (diffusers
  `enable_layerwise_casting`). That halves weight memory, but the math still runs in bf16, fp16 or fp32. Klein's official
  `fp8r` variant is presumably meant for native fp8 hardware. Whether it can be loaded and up-cast on a T4 is
  **[to verify]**. **NVFP4 needs Blackwell and is not an option.**
- int8 / 4-bit (bitsandbytes, ComfyUI int8) are **weight-only** on a T4: weights are de-quantized to the compute dtype
  inside each layer. They save memory, not compute. If the compute dtype is bf16, the matmuls are still bf16-on-Turing.

**The real options (none chosen yet)**

| # | Weights | Compute | Memory | Speed on T4 | Numerics | Spec |
|---|---|---|---|---|---|---|
| A | bf16 | bf16 (PyTorch default) | 1× | **slow** (no tensor cores) **[to measure]** | as trained | ✔ |
| B | bf16 | **fp32**, up-cast per layer | 1× weights + fp32 activations | slow (T4 fp32 ≈ 8 TFLOPS), possibly similar to A | safest | ✔ (bf16 model, fp32 math) |
| C | bf16 / 8-bit storage | **fp16** | 1× / 0.5× | **fast** (tensor cores, ~65 TFLOPS peak) | overflow risk | ✘ unless the team accepts it after a measured NaN check |
| D | 8-bit or 4-bit storage | bf16 | 0.5× / 0.3× | like A | small quantization error | ✔ |

- **We do not switch to fp16 silently.** Option C is only listed because it is the one fast path on this GPU. It would
  need an explicit team decision, plus a test that shows no NaN/inf and no visible difference against option A on the
  same seeds.
- **[estimate] why speed matters:** Klein, 16 steps, a 768² crop + 2 reference images ≈ 7k tokens, so ≈ 8·10¹³ FLOPs per
  step. At a realistic fp16 rate (~20 TFLOPS) that is ~4 s per step, about 1 min per frame. If bf16 or fp32 is 3–5× slower,
  it is **3–5 min per frame**, and 60 frames take **3–5 h**. That fits one Kaggle session, but only just. Keyframes-only
  (a spec idea), smaller crops and Qwen-Turbo's 8 steps are the levers.
- **First GPU action of M7 (≈15 min of quota):** a micro-benchmark on the T4 of the transformer's matmul and attention
  shapes in A/B/C (+ fp8-storage up-cast), then one real edit per option. Record s/step, peak VRAM and NaN check. That
  table decides the configuration.

## 5. Memory plan on 2×T4 (15 GB each, ~29 GB host RAM **[check in the session]**)

Weights only **[estimate: params × bytes]**:

| component | bf16 | 8-bit | 4-bit |
|---|---|---|---|
| Klein transformer 5.6B | 11.2 GB | 5.6 GB | ~3.2 GB |
| Klein text encoder Qwen3-VL-4B | ~8–9 GB | ~4.5 GB | ~2.5 GB |
| Klein prompt upsampler 9B (optional) | 18 GB | 9 GB | ~5 GB |
| Qwen-2.1 transformer 7B | 14 GB | 7 GB | ~4 GB |
| Qwen-2.1 text/vision encoder Qwen3-VL-8B | ~16–17 GB | ~8.5 GB | ~5 GB |

- **Klein, everything bf16 (respects the spec without quantization):** GPU 0 holds the transformer (11.2 GB + activations
  for a ≤ 1024² crop). GPU 1 holds the text encoder + VAE (fp32 per the official example; use tiling if needed). The
  prompt is the same for all frames, so **the text is encoded once**. If the encoder does not also take the images
  **[to verify]**, it can be freed after that, and the upsampler can run once at the start (4-bit, on GPU 1) and then
  be freed. BFL's 31 GB peak was measured with all parts on one GPU at 1024², so splitting across two GPUs is what
  makes this plausible **[to measure]**.
- **Qwen-2.1:** the transformer in bf16 (14 GB) leaves ~1 GB for activations on a T4, so it needs 8-bit or fp8-storage
  weights. The 8B vision-language encoder also encodes the **condition images**, so it runs for **every frame**, not once.
  It has to stay resident (8-bit, GPU 1) or be offloaded each time (slow). bf16 CPU offload of both (~30 GB) would not
  fit host RAM.
- Spec §3 suggests GPU 0 = diffusion and GPU 1 = SAM + 3DGS. Editing and rebuild are separate stages (separate
  processes, as in Week 1), so GPU 1 is free for the text encoder during editing.
- Downloads: Klein ~11 + ~9 GB (+18.8 GB upsampler). Qwen ~14 + ~17 GB. Check the Kaggle disk for the HF cache before choosing.

## 6. Recommendation and why

1. **Primary: FLUX 3 Image Klein, distilled.**
   - It is the only one of the two whose bf16 weights plausibly fit 2×T4 **without quantization**, which keeps the
     spec's precision intact.
   - It is smaller per step, the text encoder can run once per edit, it has native bounding-box hints, and an
     official FP8 storage variant.
   - Its license allows non-commercial academic use. Gated: each user accepts the license on HF and the notebook needs
     an HF token as a Kaggle Secret.
2. **Comparison: Qwen-Image-2.1-Turbo.**
   - 8 steps instead of 16 gives roughly the same transformer compute as Klein despite the 7B size **[estimate]**.
   - It mentions mask/annotation-based local edits and has transparent-layer abilities.
   - Costs: an 8B encoder that runs per frame, quantization required on a T4, and a research-only license (fine for
     coursework, but must be stated).
3. **Precision:** start with option A (bf16 weights, bf16 compute), measure option B (fp32 compute), and present option C
   (fp16) only with numbers and a NaN check. The team decides. No silent fallback: the code will **refuse** to run
   fp16 unless a flag is set explicitly, and it logs the dtype in the metrics table (deliverable 4).
4. **Mask guarantee is ours, not the model's:** crop → latent blending → hard composite, plus a unit test that every
   pixel outside the dilated mask is bit-identical.

## 7. Measurement plan (feeds deliverable 4: VRAM and timing table)

For each model and option that fits: **peak VRAM per GPU, s/step, s/frame, model load time, download size**,
quantization/dtype used. Quality protocol on 3 frames of the demo room × 2 prompts × 3 seeds:
- **outside-mask change before compositing** (PSNR of the raw model output vs the original outside the mask). This measures
  the "edits outside the mask" problem; after compositing it must be ∞ (identical).
- **colour drift across frames** (mean Lab of the edited region vs the reference edit),
- **shape consistency** (silhouette of the edited object vs the tracked mask), plus a visual grid for the report.

## 8. Open questions for the team before M7

1. Accept **Klein-first** (and Qwen-Turbo as the comparison)?
2. Precision policy: is a measured **fp16** fallback acceptable at all if bf16/fp32 turn out too slow, or is it strictly
   bf16/fp32 and we reduce work instead (keyframes, smaller crops)?
3. Who accepts the Klein license on Hugging Face and provides an HF token for the Kaggle Secret?
4. Person A: the mask format (proposed earlier: one PNG per frame, 0/255, frame size) and whether masks come with a
   per-frame "object present" flag (needed to skip frames).

## Sources
- BFL docs, FLUX 3 Image Klein: https://docs.bfl.ml/flux_3/flux3_image_klein
- BFL release notes: https://docs.bfl.ml/release-notes
- HF model cards: https://huggingface.co/black-forest-labs/flux-3-image-klein · https://huggingface.co/black-forest-labs/flux-3-image-klein-base
- Qwen-Image-2.1 GitHub: https://github.com/QwenLM/Qwen-Image-2.1 · license: https://github.com/QwenLM/Qwen-Image-2.1/blob/main/LICENSE
- HF model cards: https://huggingface.co/Qwen/Qwen-Image-2.1 · https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo · https://huggingface.co/Comfy-Org/Qwen-Image-2.1
- bf16 on T4: https://earino.github.io/applied-deep-learning/site/blog/05_bf16_on_t4.html ·
  https://discuss.pytorch.org/t/current-cuda-device-does-not-support-bfloat16-please-switch-dtype-to-float16/201564 ·
  https://huggingface.co/asdf98/iris-image-gen/commit/e90110a4b3cfc4de230ca03175b6bbc5cb94fa30
- fp8 storage + up-cast (layerwise casting), T4 numbers for FLUX.1 (proposal, unverified): https://github.com/huggingface/diffusers/pull/9177 ·
  https://github.com/invoke-ai/InvokeAI/pull/8945
- Secondary (context only): https://www.eesel.ai/blog/flux-3-klein · https://datanorth.ai/news/qwen-releases-qwen-image-2-1
