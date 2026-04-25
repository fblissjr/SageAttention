#!/usr/bin/env python3
"""Image-gen self-attn shapes for sage on sm89.

Companion to `test_sageattn_ltx_shapes.py`. Same harness, same dispatchers,
different shape catalog. Carries shapes for image-gen model classes
(non-LTX head_dims) so the LTX file stays focused and fast.

Two shapes today:

- **Flux-class** self-attn at 1024^2 output. 1024^2 / 16^2 VAE = 4096
  image tokens; head_dim=128, heads=24 are the Flux-1-dev family
  defaults. Confirms sage's speedup holds on head_dim=128 workloads,
  not just LTX's head_dim=64.
- **Z-Image-Turbo** self-attn (S3-DiT, single-stream). Architecture:
  30 layers, hidden=3840, 32 heads, head_dim=120 (3840/32). Single-
  stream means text+image tokens concatenate into one sequence;
  ~4096 image tokens + ~512 text tokens ~= 4608. head_dim=120 is
  non-power-of-2; this row would SKIP if sage's CUDA kernels can't
  handle it (verified 2026-04-25: they do).

Run from the venv with sage installed:
    ${VIRTUAL_ENV}/bin/python tests/test_sageattn_image_shapes.py

Adds an image-gen row by editing `IMAGE_SHAPES` below, not the LTX file.
"""

from __future__ import annotations

# Reuse the LTX file's helpers verbatim. The split is purely about which
# shapes belong to which model class -- the dispatch and measurement
# scaffolding is identical.
from test_sageattn_ltx_shapes import (
    FLASHINFER_MODE_SPECS,
    FP8PP_LABEL,
    MODE_SPECS,
    Shape,
    SPARGE_MODE_SPECS,
    TORCH_MODE_SPECS,
    TRITON_LABEL,
    accuracy_metrics,
    build_padding_mask,
    dispatch_flashinfer,
    dispatch_sage,
    dispatch_sparge,
    dispatch_torch,
    make_qkv,
    measure_mode,
    print_header,
    sdpa_reference,
    time_median_ms,
)

import torch


IMAGE_SHAPES = [
    # Flux-class self-attn at 1024^2.
    Shape("image_gen_self_attn_4096_h24_d128", 1, 24, 4096, 4096, 128, False),
    # Z-Image-Turbo (S3-DiT) self-attn. head_dim=120 is the unusual one.
    Shape("z_image_turbo_self_attn_4608_h32_d120", 1, 32, 4608, 4608, 120, False),
]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available -- this test measures kernel numerics on-GPU.")
        return

    label_width = max(
        len(spec[0]) for spec in
        (*MODE_SPECS, *TORCH_MODE_SPECS, *FLASHINFER_MODE_SPECS, *SPARGE_MODE_SPECS)
    )
    print_header(label_width)

    dtype = torch.bfloat16
    warnings: list[str] = []

    for shape in IMAGE_SHAPES:
        print()
        print(f"=== {shape.name} ===")
        print(
            f"    B={shape.batch} H={shape.heads} Sq={shape.seq_q} Skv={shape.seq_kv} "
            f"D={shape.head_dim} mask={shape.has_mask} dtype={dtype}"
        )
        q, k, v = make_qkv(shape, dtype, v_std=shape.v_std)
        mask = build_padding_mask(shape) if shape.has_mask else None

        out_ref = sdpa_reference(q, k, v, mask)
        sdpa_ms = time_median_ms(lambda: sdpa_reference(q, k, v, mask))
        print(f"    {'SDPA (math)':<{label_width}}"
              f"  {'-':>10}  {'-':>10}  {'-':>10}  {'-':>10}  {sdpa_ms:>10.2f}  {1.0:>5.2f}x")

        def _print_row(label, mean_r, max_r, mean_a, max_a, median_ms, warn_threshold=0.10, warn=True):
            marker = "  !" if warn and mean_r > warn_threshold else ""
            ms_cell = f"{median_ms:>10.2f}" if median_ms is not None else f"{'-':>10}"
            speed_cell = f"{sdpa_ms / median_ms:>5.2f}x" if median_ms is not None else f"{'-':>5} "
            print(
                f"    {label:<{label_width}}  "
                f"{mean_r:>10.3g}  {max_r:>10.3g}  "
                f"{mean_a:>10.3g}  {max_a:>10.3g}  "
                f"{ms_cell}  {speed_cell}{marker}"
            )
            if warn and mean_r > warn_threshold:
                warnings.append(f"{shape.name} / {label}: mean_rtol={mean_r:.3g}")

        def _print_result(label, m, warn_rtol=True):
            _print_row(label, m.mean_rtol, m.max_rtol, m.mean_atol, m.max_atol,
                       median_ms=m.median_ms, warn=warn_rtol)

        cached_outs = {}
        for label, kernel_name, kwargs in MODE_SPECS:
            try:
                mode_fn = dispatch_sage(kernel_name, kwargs)
                m, out = measure_mode(mode_fn, q, k, v, mask, out_ref)
            except Exception as exc:
                print(f"    {label:<{label_width}}  SKIP ({type(exc).__name__}: {str(exc)[:80]})")
                continue
            _print_result(label, m)
            if label in (TRITON_LABEL, FP8PP_LABEL):
                cached_outs[label] = out

        if not shape.has_mask and TRITON_LABEL in cached_outs and FP8PP_LABEL in cached_outs:
            mean_r, max_r, mean_a, max_a = accuracy_metrics(
                cached_outs[FP8PP_LABEL], cached_outs[TRITON_LABEL]
            )
            _print_row("fp8++vs.triton", mean_r, max_r, mean_a, max_a,
                       median_ms=None, warn_threshold=0.15)

        def _run_aux(specs, dispatch_factory, warn_rtol):
            for label, payload in specs:
                try:
                    mode_fn = dispatch_factory(payload)
                    mm, _ = measure_mode(mode_fn, q, k, v, mask, out_ref)
                except Exception as exc:
                    print(f"    {label:<{label_width}}  SKIP ({type(exc).__name__}: {str(exc)[:80]})")
                    continue
                _print_result(label, mm, warn_rtol=warn_rtol)

        _run_aux(TORCH_MODE_SPECS, dispatch_torch, warn_rtol=False)
        _run_aux(FLASHINFER_MODE_SPECS, dispatch_flashinfer, warn_rtol=False)
        _run_aux(SPARGE_MODE_SPECS, dispatch_sparge, warn_rtol=True)

    print()
    if warnings:
        print(f"Soft warnings ({len(warnings)}): mean_rtol > 0.10 on:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("All (shape, mode) pairs: mean_rtol <= 0.10.")


if __name__ == "__main__":
    main()
