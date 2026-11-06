#!/usr/bin/env python3
"""T4 precision micro-benchmark (research §4, "first GPU action of M7", ~2 min of GPU).

Times the two operations that dominate a diffusion transformer step: a linear layer and attention, at shapes
close to Klein's (hidden 3072, 24 heads x 128, ~7k tokens for a 768^2 crop + 2 references), in each precision
option. Prints a table and writes JSON. It measures the *hardware path*, not image quality: quality and NaNs of the
real model are checked by run_edit.py's report (degenerate flag) on the same seeds.

  python backend/editing/bench_t4.py --out /kaggle/working/bench_t4.json
"""
from __future__ import annotations

import argparse
import json
import time


def timeit(fn, warmup=2, reps=5):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps, out


def main(argv=None):
    import torch
    import torch.nn.functional as F
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=7168)
    ap.add_argument("--hidden", type=int, default=3072)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--out", default="bench_t4.json")
    a = ap.parse_args(argv)
    dev = "cuda:0"
    cap = torch.cuda.get_device_capability(0)
    res = {"gpu": torch.cuda.get_device_name(0), "capability": f"{cap[0]}.{cap[1]}", "torch": torch.__version__,
           "is_bf16_supported_claim": torch.cuda.is_bf16_supported(), "shapes": vars(a), "rows": []}
    T, D, Hh = a.tokens, a.hidden, a.heads
    hd = D // Hh
    x32 = torch.randn(T, D, device=dev) * 4  # activations with some range, to show fp16 overflow risk
    w32 = torch.randn(4 * D, D, device=dev) / D ** 0.5
    q32 = torch.randn(1, Hh, T, hd, device=dev)
    flops_lin = 2 * T * D * 4 * D
    flops_att = 4 * Hh * T * T * hd

    def row(option, name, lin, att):
        try:
            t_lin, y = timeit(lin)
            t_att, z = timeit(att)
            bad = bool(torch.isnan(y).any() or torch.isinf(y).any() or torch.isnan(z).any() or torch.isinf(z).any())
            r = {"option": option, "path": name, "linear_ms": round(t_lin * 1e3, 2), "attention_ms": round(t_att * 1e3, 2),
                 "linear_tflops": round(flops_lin / t_lin / 1e12, 2), "attention_tflops": round(flops_att / t_att / 1e12, 2),
                 "nan_or_inf": bad, "error": None}
        except Exception as e:  # e.g. a kernel that does not exist for this dtype on Turing
            r = {"option": option, "path": name, "error": f"{e.__class__.__name__}: {e}"[:300]}
        res["rows"].append(r)
        print(r, flush=True)

    bf, f16, f32 = torch.bfloat16, torch.float16, torch.float32
    xb, wb, qb = x32.to(bf), w32.to(bf), q32.to(bf)
    xh, wh, qh = x32.to(f16), w32.to(f16), q32.to(f16)
    row("A", "bf16 weights, bf16 compute", lambda: F.linear(xb, wb), lambda: F.scaled_dot_product_attention(qb, qb, qb))
    row("B", "bf16 weights, fp32 compute (up-cast per layer)", lambda: F.linear(x32, wb.to(f32)),
        lambda: F.scaled_dot_product_attention(q32, q32, q32))
    row("C", "fp16 weights, fp16 compute (NOT the spec's dtype)", lambda: F.linear(xh, wh),
        lambda: F.scaled_dot_product_attention(qh, qh, qh))
    if hasattr(torch, "float8_e4m3fn"):
        w8 = w32.to(torch.float8_e4m3fn)
        row("D", "fp8 storage, up-cast to bf16 per layer", lambda: F.linear(xb, w8.to(bf)),
            lambda: F.scaled_dot_product_attention(qb, qb, qb))
    # overflow probe: large activations, as they occur in some DiT layers
    big = torch.full((256, D), 300.0, device=dev)
    for dt, nm in ((bf, "bf16"), (f16, "fp16")):
        y = F.linear(big.to(dt), torch.ones(D, D, device=dev, dtype=dt))
        res[f"overflow_probe_{nm}"] = "inf" if torch.isinf(y).any().item() else "ok"
    print({k: v for k, v in res.items() if k.startswith("overflow")})
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
