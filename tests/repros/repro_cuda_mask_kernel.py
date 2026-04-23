#!/usr/bin/env python3
"""Reference + regression test for the CUDA mask-path defect.

sage's CUDA kernels (`_qattn_sm80`, `_qattn_sm89`) have no general
attention-mask support. The Python wrappers `sageattn_qk_int8_pv_fp16_cuda`
and `sageattn_qk_int8_pv_fp8_cuda` accept `attn_mask` via **kwargs but
never pass it through; the C++ kernels' MaskMode enum only has {kNone,
kCausal}. The mask is silently dropped, producing wrong output whose
rtol grows as (masked_positions / total_positions).

This isn't a bug to patch — it's a missing feature to add. See
CHANGELOG.md "Known kernel bugs" and "Open work". Consumer-side
workaround lives in a downstream ComfyUI node that routes masked
calls to the triton path. Kernel-side fix is backlog.

Affected kernels:
    sageattn_qk_int8_pv_fp16_cuda  (pv_accum_dtype="fp32")
    sageattn_qk_int8_pv_fp8_cuda   (pv_accum_dtype="fp32+fp32")
    sageattn_qk_int8_pv_fp8_cuda   (pv_accum_dtype="fp32+fp16", aka "++")

Triton path (`sageattn_qk_int8_pv_fp16_triton`) is correct (has proper
mask plumbing).

Measured on RTX 4090 (sm89), CUDA 13.2, torch 2.11.0:

    seq_kv  fp16_cuda  fp8_cuda  fp8++  fp16_triton
    ------  ---------  --------  -----  -----------
        32      NaN       0.941  0.941       0.032
        64      0.684     0.685  0.685       0.039
       128      0.530     0.530  0.530       0.040
       226      0.439     0.440  0.440       0.039
       512      0.332     0.335  0.335       0.040
      1024      0.264     0.268  0.269       0.040

max_atol at seq_kv=226 hits 1.27 — individual outputs disagree with
the reference by more than a full unit. At seq_kv=32 with proportionally
small pad_tail, fp16_cuda returns NaN outright. NaN mechanism: when most
large logits land in masked positions, the per-block max is computed over
poisoned values; exp(score - max) underflows to 0 for unmasked positions;
softmax denominator → 0; division → NaN.

Run:
    ${VIRTUAL_ENV}/bin/python tests/repros/repro_cuda_mask_kernel.py
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sageattention import (
    sageattn_qk_int8_pv_fp16_cuda,
    sageattn_qk_int8_pv_fp16_triton,
    sageattn_qk_int8_pv_fp8_cuda,
)


def rtol_max_atol(actual: torch.Tensor, expect: torch.Tensor) -> tuple[float, float, float]:
    a, e = actual.float(), expect.float()
    diff = (a - e).abs()
    eps = torch.finfo(a.dtype).eps
    rdiff = diff / torch.maximum(torch.maximum(a.abs(), e.abs()), torch.tensor(eps, device=a.device))
    return rdiff.mean().item(), rdiff.max().item(), diff.max().item()


def run(seq_kv: int) -> None:
    torch.manual_seed(0)
    B, H, Sq, D = 1, 32, 31776, 64
    dtype = torch.bfloat16
    q = torch.randn(B, H, Sq,     D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, seq_kv, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, seq_kv, D, device="cuda", dtype=dtype)

    # Bool mask: last 30 kv positions masked out (typical text-padding pattern).
    pad_tail = min(30, seq_kv // 2)
    mask = torch.ones(B, H, Sq, seq_kv, device="cuda", dtype=torch.bool)
    mask[..., -pad_tail:] = False

    # Reference: torch SDPA efficient-attention backend (flash/memory-efficient,
    # O(N) memory -- MATH backend OOMs at this seq_q).
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def report(label: str, fn):
        try:
            out = fn()
        except Exception as exc:
            print(f"  {label:<40} SKIP ({type(exc).__name__}: {str(exc)[:60]})")
            return
        mean_r, max_r, max_a = rtol_max_atol(out, out_ref)
        marker = "  <-- BUG" if mean_r > 0.10 else ""
        print(f"  {label:<40} mean_rtol={mean_r:7.3f}  max_rtol={max_r:7.3f}  max_atol={max_a:7.4f}{marker}")

    print(f"seq_kv={seq_kv} (B={B} H={H} Sq={Sq} D={D} dtype={dtype}, bool mask, pad_tail={pad_tail}):")
    report("sageattn_qk_int8_pv_fp16_cuda",
        lambda: sageattn_qk_int8_pv_fp16_cuda(q, k, v, attn_mask=mask, is_causal=False,
                                              pv_accum_dtype="fp32", tensor_layout="HND"))
    report("sageattn_qk_int8_pv_fp8_cuda",
        lambda: sageattn_qk_int8_pv_fp8_cuda(q, k, v, attn_mask=mask, is_causal=False,
                                             pv_accum_dtype="fp32+fp32", tensor_layout="HND"))
    report("sageattn_qk_int8_pv_fp8_cuda (++)",
        lambda: sageattn_qk_int8_pv_fp8_cuda(q, k, v, attn_mask=mask, is_causal=False,
                                             pv_accum_dtype="fp32+fp16", tensor_layout="HND"))
    report("sageattn_qk_int8_pv_fp16_triton",
        lambda: sageattn_qk_int8_pv_fp16_triton(q, k, v, attn_mask=mask, is_causal=False,
                                                tensor_layout="HND"))
    print()


def main() -> None:
    import sageattention
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}")
    print(f"sage:   {getattr(sageattention, '__version__', '?')}")
    print()
    for seq_kv in (32, 64, 128, 226, 512, 1024):
        run(seq_kv)


if __name__ == "__main__":
    main()
