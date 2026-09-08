#!/usr/bin/env python3
"""Where a MiniMax H3 DiT block spends time AND peak memory, per sub-module.

## Why this exists

LTX has `docs/ltx_workload_profile.md` -- sub-module shares from a real render,
the canonical input for ranking a perf bet. H3 has had no equivalent, and two
claims that rank all H3 work were resting on nothing this specific:

- **"attention is almost all of it"**, which turned out to be 76% measured at
  a sequence length past H3's legal ceiling. A bound from a render A/B puts
  attention at **>= ~32%** of a real render with this fork
  (`docs/h3_attention_stack.md`). This file gives the within-block split that
  bound cannot.
- **that memory work belongs in the attention kernel.** The consumer patches
  `optimized_attention`, which receives *finished* q/k/v, so every headroom
  lever this fork has -- `sageattn_consume`, the caller-side `v` clone, the
  `per_channel_fp8` transpose buffer -- acts after the projection has already
  allocated. A producer-side approach that chunks the projection output
  reportedly saves gigabytes where ours save hundreds of MiB. If the peak is
  not where we have been working, that is worth knowing before another kernel
  day goes into it.

So this reports **time and peak transient per sub-module**, because on this
model memory decides whether a render happens at all and time only decides how
long it takes.

## What this is and is not

Synthetic weights at the real config, one block, no sampler and no render.
Legitimate for what it measures: input distribution does not change the work
done, so speed and allocation are fair on random data (see
`docs/testing_practices.md` for the three-way split).

**It is isolation evidence -- Cell A/B, not delivered.** A block measured
alone cannot see L2 contention with neighbouring modules, allocator state
across a real step, or offload behaviour. This repo has been burned by exactly
that: a primitive that benched 1.26-1.36x in isolation came back +1.79% slower
in-pipeline. Read the shares, not the absolute times, and confirm anything
load-bearing against a real render.

    ${VIRTUAL_ENV}/bin/python tests/bench_h3_block_profile.py
    ${VIRTUAL_ENV}/bin/python tests/bench_h3_block_profile.py --seq 109126
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn

# comfy/ldm/minimax/model.py:474-477, read 2026-09-08.
HIDDEN, LAYERS, HEADS, HEAD_DIM, FFN = 5376, 50, 56, 128, 14336
INNER = HEADS * HEAD_DIM  # 7168; note INNER != HIDDEN on this model

# 1344x768, 124 frames, two keyframes -- the packed length the gated bench row
# uses and the shape the consumer's default graphs render.
DEFAULT_SEQ = 41822


def _timed(fn, warmup=2, iters=5):
    """Median ms and peak transient MiB for one callable.

    Peak is measured against the allocator's state on entry, so it is the
    transient this sub-module ADDS rather than the process peak -- which is the
    number that decides whether moving a seam would help.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
        del out
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    times.sort()
    return times[len(times) // 2], peak


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    ap.add_argument("--weights", choices=("bf16", "int8"), default="bf16",
                    help="int8 mirrors production: the consumer's H3 base is "
                         "INT8 ConvRot, so every Linear goes through "
                         "comfy_kitchen.int8_linear and fc2 folds the SwiGLU "
                         "into its input quantizer. bf16 is plain nn.Linear "
                         "and does NOT describe a real render.")
    args = ap.parse_args()
    S = args.seq

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    from sageattention import sageattn, build_info

    dev, dt = "cuda", torch.bfloat16
    print(f"H3 DiT block profile -- S={S:,}  hidden={HIDDEN} heads={HEADS} "
          f"head_dim={HEAD_DIM} ffn={FFN}")
    print(f"sage: {build_info()['describe']}  torch: {torch.__version__}  "
          f"weights: {args.weights}"
          + ("" if args.weights == "int8" else "  <- NOT production; see --weights int8"))
    print(f"one block; the model stacks {LAYERS} identical ones, so shares "
          f"carry and absolute times multiply.\n")

    if args.weights == "int8":
        import comfy_kitchen as ck

        def q(w):
            qd, sc = ck.quantize_int8_tensorwise(w)
            return qd, sc

        def mk(i, o):
            return q(torch.randn(o, i, device=dev, dtype=dt) * 0.02)

        Wqkv, Wout = mk(HIDDEN, INNER * 3), mk(INNER, HIDDEN)
        W1, W2 = mk(HIDDEN, FFN * 2), mk(FFN, HIDDEN)
        lin = lambda W, x, act=None: ck.int8_linear(
            x, W[0], W[1], None, dt, convrot=False, input_act=act)

    qkv_proj = nn.Linear(HIDDEN, INNER * 3, bias=False, device=dev, dtype=dt)
    out_proj = nn.Linear(INNER, HIDDEN, bias=False, device=dev, dtype=dt)
    fc1 = nn.Linear(HIDDEN, FFN * 2, bias=False, device=dev, dtype=dt)
    fc2 = nn.Linear(FFN, HIDDEN, bias=False, device=dev, dtype=dt)
    norm = nn.RMSNorm(HIDDEN, device=dev, dtype=dt)
    x = torch.randn(S, HIDDEN, device=dev, dtype=dt)

    # Each stage builds only its own inputs and frees them before the next.
    # Building all of them up front held ~19 GB at the ceiling length and OOMed
    # every interesting stage -- a property of the harness, not the model.
    def _q_kv():
        qkv = qkv_proj(x)
        out = tuple(t.view(S, HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0).contiguous()
                    for t in qkv.split(INNER, dim=-1))
        del qkv
        return out

    int8 = args.weights == "int8"
    stages = [
        ("norm (RMSNorm x2)", lambda: ((), lambda _: (norm(x), norm(x)))),
        (f"qkv_proj  {'int8' if int8 else 'Linear'}",
         lambda: ((), (lambda _: lin(Wqkv, x)) if int8 else (lambda _: qkv_proj(x)))),
        ("attention sage",
         lambda: (_q_kv(), lambda t: sageattn(t[0], t[1], t[2],
                                              tensor_layout="HND", is_causal=False))),
        (f"out_proj  {'int8' if int8 else 'Linear'}",
         lambda: ((torch.randn(S, INNER, device=dev, dtype=dt),),
                  (lambda t: lin(Wout, t[0])) if int8 else (lambda t: out_proj(t[0])))),
        (f"mlp fc1   {'int8' if int8 else 'Linear'}",
         lambda: ((), (lambda _: lin(W1, x)) if int8 else (lambda _: fc1(x)))),
        # Production calls fc2 on fc1's FULL 2*ffn output with the SwiGLU folded
        # into its input quantizer; the bf16 arm has no such fusion and takes
        # the post-activation tensor instead.
        (f"mlp fc2   {'int8+swiglu' if int8 else 'Linear'}",
         lambda: ((torch.randn(S, FFN * (2 if int8 else 1), device=dev, dtype=dt),),
                  (lambda t: lin(W2, t[0], "swiglu")) if int8 else (lambda t: fc2(t[0])))),
    ]

    rows = []
    with torch.inference_mode():
        for label, build in stages:
            try:
                inputs, call = build()
                ms, mib = _timed(lambda: call(inputs))
                rows.append((label, ms, mib))
                del inputs, call
            except Exception as exc:  # noqa: BLE001
                print(f"  {label:<22} FAILED: {type(exc).__name__}: {str(exc)[:60]}")
            torch.cuda.empty_cache()

    tot_ms = sum(r[1] for r in rows)
    print(f"{'sub-module':<22}{'ms':>9}{'% time':>9}{'peak MiB':>11}")
    for label, ms, mib in rows:
        print(f"{label:<22}{ms:>9.2f}{100*ms/tot_ms:>8.1f}%{mib:>11.0f}")
    print(f"{'TOTAL (block)':<22}{tot_ms:>9.2f}{100:>8.1f}%")

    attn_ms = next((r[1] for r in rows if r[0].startswith("attention")), 0.0)
    peak_row = max(rows, key=lambda r: r[2]) if rows else None
    print()
    print(f"attention is {100*attn_ms/tot_ms:.0f}% of block compute here. That is an "
          f"isolation number:")
    print(f"  it excludes adaln, rope, the modulation adds, sampler overhead, "
          f"VAE decode and offload,")
    print(f"  all of which sit in a render's denominator. The render-level "
          f"bound is >= ~32%.")
    if peak_row:
        print(f"\nlargest single transient: {peak_row[0].strip()} at "
              f"{peak_row[2]:.0f} MiB.")
        print("  If that is not the attention kernel, this fork's headroom "
              "levers are aimed off-target:")
        print("  they all act after the projection, and cannot reach an "
              "allocation made before it.")


if __name__ == "__main__":
    main()
