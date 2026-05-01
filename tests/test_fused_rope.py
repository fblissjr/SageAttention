#!/usr/bin/env python3
"""Validate the fused split-RoPE primitive (`sageattention.fused_rope_split`).

Three layers:
1. CPU-only: my torch-reference implementation matches an inlined
   equivalent of LTX's `apply_split_rotary_emb` (comfy/ldm/lightricks/
   model.py:343). Catches indexing-math bugs without touching the GPU.
2. GPU: the triton kernel matches the torch reference within bf16 / fp16
   precision floor (rtol ~ 1e-3) on LTX video (D=128) and audio (D=64)
   shapes.
3. GPU: API contracts -- in-place semantics, fallback path, dtype
   guards, freqs_cis tuple variants.

Run:
    ${VIRTUAL_ENV}/bin/python tests/test_fused_rope.py
"""

import sys
import traceback

import torch

from sageattention.triton.fused_rope import (
    _torch_split_rope_reference,
    fused_rope_split,
)


def _ltx_reference_apply_split_rotary_emb(
    input_tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    """Inline equivalent of `comfy/ldm/lightricks/model.py::apply_split_rotary_emb`,
    rewritten without einops so the test has zero external deps.

    Verified by-hand against the LTX source: split last dim into halves
    `first` and `second`, then `out_first = first*cos - second*sin`,
    `out_second = second*cos + first*sin`. Returns same shape as input.
    """
    needs_reshape = False
    if input_tensor.ndim != 4 and cos.ndim == 4:
        B, H, T, _ = cos.shape
        input_tensor = input_tensor.reshape(B, T, H, -1).swapaxes(1, 2)
        needs_reshape = True

    D = input_tensor.shape[-1]
    D_half = D // 2
    first = input_tensor[..., :D_half].clone()
    second = input_tensor[..., D_half:].clone()
    cos_b = cos.unsqueeze(-2) if cos.ndim == input_tensor.ndim else cos
    sin_b = sin.unsqueeze(-2) if sin.ndim == input_tensor.ndim else sin
    cos_b = cos_b.squeeze(-2) if cos_b.ndim == input_tensor.ndim + 1 else cos_b
    sin_b = sin_b.squeeze(-2) if sin_b.ndim == input_tensor.ndim + 1 else sin_b
    out_first = first * cos_b - second * sin_b
    out_second = second * cos_b + first * sin_b
    out = torch.cat([out_first, out_second], dim=-1)

    if needs_reshape:
        B, H, T, D = out.shape
        out = out.swapaxes(1, 2).reshape(B, T, H * D)
    return out


def _make_pe(B: int, H: int, T: int, D: int, dtype: torch.dtype, device: str):
    """Build a (cos, sin) RoPE encoding shaped [B, H, T, D//2] with values
    that exercise both magnitudes (cos != 1 and sin != 0 across positions).
    Uses a small angle progression so accumulated rotation stays bounded."""
    freqs = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(-1)
    freqs = freqs * torch.linspace(0.01, 0.5, D // 2, device=device, dtype=torch.float32).unsqueeze(0)
    cos = freqs.cos().to(dtype)[None, None, :, :].expand(B, H, T, D // 2).contiguous()
    sin = freqs.sin().to(dtype)[None, None, :, :].expand(B, H, T, D // 2).contiguous()
    return cos, sin


def _accuracy(actual: torch.Tensor, expect: torch.Tensor) -> tuple[float, float]:
    a = actual.float()
    e = expect.float()
    diff = (a - e).abs()
    eps = torch.tensor(torch.finfo(a.dtype).eps, device=a.device, dtype=a.dtype)
    rdiff = diff / torch.maximum(torch.maximum(a.abs(), e.abs()), eps)
    return rdiff.mean().item(), rdiff.max().item()


# --- Tests --------------------------------------------------------------

def test_torch_reference_matches_ltx_apply_split_rotary_emb_cpu():
    """My _torch_split_rope_reference produces the same output as the
    inlined LTX reference. Pure CPU; no GPU contention."""
    B, H, T, D = 2, 4, 16, 64
    x = torch.randn(B, T, H, D, dtype=torch.float32)
    cos, sin = _make_pe(B, H, T, D, torch.float32, "cpu")
    x_hnd = x.transpose(1, 2).contiguous()
    out_mine = _torch_split_rope_reference(x_hnd, cos, sin)
    out_ltx = _ltx_reference_apply_split_rotary_emb(x_hnd, cos, sin)
    mean_r, max_r = _accuracy(out_mine, out_ltx)
    assert max_r < 1e-6, f"CPU reference mismatch: max_rtol={max_r}"


def test_torch_reference_handles_3d_input_cpu():
    """The reference accepts the [B, T, H*D] flat layout that LTX sees
    right after the linear projection (pre-reshape)."""
    B, H, T, D = 2, 4, 16, 64
    x_flat = torch.randn(B, T, H * D, dtype=torch.float32)
    cos, sin = _make_pe(B, H, T, D, torch.float32, "cpu")
    out_mine = _torch_split_rope_reference(x_flat, cos, sin)
    out_ltx = _ltx_reference_apply_split_rotary_emb(x_flat, cos, sin)
    mean_r, max_r = _accuracy(out_mine, out_ltx)
    assert max_r < 1e-6, f"CPU 3d-input reference mismatch: max_rtol={max_r}"


def test_triton_matches_reference_ltx_video_d128_bf16():
    """LTX video self-attn shape (D=128) at bf16. rtol within bf16 floor."""
    if not torch.cuda.is_available():
        return ("SKIP", "no CUDA")
    B, H, T, D = 1, 32, 256, 128
    dtype = torch.bfloat16
    cos, sin = _make_pe(B, H, T, D, dtype, "cuda")
    q = torch.randn(B, T, H * D, dtype=dtype, device="cuda")
    k = torch.randn(B, T, H * D, dtype=dtype, device="cuda")
    q_ref = _torch_split_rope_reference(q.clone(), cos, sin)
    k_ref = _torch_split_rope_reference(k.clone(), cos, sin)
    q_out, k_out = fused_rope_split(q, k, (cos, sin, True))
    q_mr, q_xr = _accuracy(q_out, q_ref)
    k_mr, k_xr = _accuracy(k_out, k_ref)
    # bf16 mantissa ~7 bits -> rtol floor ~ 1e-2 in worst case; mean should be << that.
    assert q_mr < 5e-3, f"q mean_rtol too high: {q_mr}"
    assert k_mr < 5e-3, f"k mean_rtol too high: {k_mr}"


def test_triton_matches_reference_ltx_audio_d64_bf16():
    """LTX audio path (D=64). Same harness as D=128 test."""
    if not torch.cuda.is_available():
        return ("SKIP", "no CUDA")
    B, H, T, D = 1, 32, 256, 64
    dtype = torch.bfloat16
    cos, sin = _make_pe(B, H, T, D, dtype, "cuda")
    q = torch.randn(B, T, H * D, dtype=dtype, device="cuda")
    k = torch.randn(B, T, H * D, dtype=dtype, device="cuda")
    q_ref = _torch_split_rope_reference(q.clone(), cos, sin)
    k_ref = _torch_split_rope_reference(k.clone(), cos, sin)
    q_out, k_out = fused_rope_split(q, k, (cos, sin, True))
    q_mr, _ = _accuracy(q_out, q_ref)
    k_mr, _ = _accuracy(k_out, k_ref)
    assert q_mr < 5e-3 and k_mr < 5e-3, f"D=64 mean_rtol q={q_mr} k={k_mr}"


def test_triton_matches_reference_fp16():
    """fp16 dtype path. Same math, IS_BF16=False store branch."""
    if not torch.cuda.is_available():
        return ("SKIP", "no CUDA")
    B, H, T, D = 1, 8, 128, 64
    dtype = torch.float16
    cos, sin = _make_pe(B, H, T, D, dtype, "cuda")
    q = torch.randn(B, T, H * D, dtype=dtype, device="cuda")
    k = torch.randn(B, T, H * D, dtype=dtype, device="cuda")
    q_ref = _torch_split_rope_reference(q.clone(), cos, sin)
    k_ref = _torch_split_rope_reference(k.clone(), cos, sin)
    q_out, k_out = fused_rope_split(q, k, (cos, sin, True))
    q_mr, _ = _accuracy(q_out, q_ref)
    k_mr, _ = _accuracy(k_out, k_ref)
    # fp16 mantissa ~10 bits -> tighter than bf16
    assert q_mr < 1e-3 and k_mr < 1e-3, f"fp16 mean_rtol q={q_mr} k={k_mr}"


def test_use_triton_false_uses_reference_path():
    """use_triton=False forces the torch fallback. Output must match the
    reference exactly (bit-equivalent; same code path)."""
    if not torch.cuda.is_available():
        return ("SKIP", "no CUDA")
    B, H, T, D = 1, 8, 64, 64
    dtype = torch.bfloat16
    cos, sin = _make_pe(B, H, T, D, dtype, "cuda")
    q = torch.randn(B, T, H * D, dtype=dtype, device="cuda")
    k = torch.randn(B, T, H * D, dtype=dtype, device="cuda")
    q_ref = _torch_split_rope_reference(q.clone(), cos, sin)
    k_ref = _torch_split_rope_reference(k.clone(), cos, sin)
    q_out, k_out = fused_rope_split(q, k, (cos, sin, True), use_triton=False)
    assert torch.equal(q_out, q_ref), "use_triton=False q output diverged from reference"
    assert torch.equal(k_out, k_ref), "use_triton=False k output diverged from reference"


def test_2tuple_freqs_cis_defaults_split_pe_true():
    """A 2-tuple `(cos, sin)` (no explicit split_pe flag) must still
    take the triton path -- absence of the flag means split-pe per
    LTX's apply_rotary_emb default."""
    if not torch.cuda.is_available():
        return ("SKIP", "no CUDA")
    B, H, T, D = 1, 4, 32, 64
    dtype = torch.bfloat16
    cos, sin = _make_pe(B, H, T, D, dtype, "cuda")
    q = torch.randn(B, T, H * D, dtype=dtype, device="cuda")
    k = torch.randn(B, T, H * D, dtype=dtype, device="cuda")
    q_ref = _torch_split_rope_reference(q.clone(), cos, sin)
    q_out, k_out = fused_rope_split(q, k, (cos, sin))
    mr, _ = _accuracy(q_out, q_ref)
    assert mr < 5e-3, f"2-tuple freqs_cis path mean_rtol={mr}"


def test_split_pe_false_falls_back_to_torch():
    """split_pe=False signals the interleaved variant; v1 falls back
    to torch (the ref path also handles split-only, so this validates
    the precondition guard)."""
    if not torch.cuda.is_available():
        return ("SKIP", "no CUDA")
    B, H, T, D = 1, 4, 32, 64
    dtype = torch.bfloat16
    cos, sin = _make_pe(B, H, T, D, dtype, "cuda")
    q = torch.randn(B, T, H * D, dtype=dtype, device="cuda")
    k = torch.randn(B, T, H * D, dtype=dtype, device="cuda")
    q_in = q.clone()
    q_out, k_out = fused_rope_split(q, k, (cos, sin, False))
    # Fallback creates a NEW tensor; q's original buffer not mutated.
    assert torch.equal(q, q_in), "split_pe=False fallback mutated q in place"


def test_triton_path_mutates_in_place():
    """Triton fast path returns the same buffers passed in (in-place
    write). Documented contract; consumers must not re-use the input."""
    if not torch.cuda.is_available():
        return ("SKIP", "no CUDA")
    B, H, T, D = 1, 4, 32, 64
    dtype = torch.bfloat16
    cos, sin = _make_pe(B, H, T, D, dtype, "cuda")
    q = torch.randn(B, T, H * D, dtype=dtype, device="cuda")
    k = torch.randn(B, T, H * D, dtype=dtype, device="cuda")
    q_id_before = q.data_ptr()
    k_id_before = k.data_ptr()
    q_out, k_out = fused_rope_split(q, k, (cos, sin, True))
    assert q_out.data_ptr() == q_id_before, "triton path did not return q in place"
    assert k_out.data_ptr() == k_id_before, "triton path did not return k in place"


def test_invalid_dtype_raises():
    """fp32 q/k is not supported (sage's downstream quant kernels assume
    bf16/fp16). Must raise, not silently produce wrong output."""
    if not torch.cuda.is_available():
        return ("SKIP", "no CUDA")
    B, H, T, D = 1, 4, 32, 64
    cos, sin = _make_pe(B, H, T, D, torch.float32, "cuda")
    q = torch.randn(B, T, H * D, dtype=torch.float32, device="cuda")
    k = torch.randn(B, T, H * D, dtype=torch.float32, device="cuda")
    try:
        fused_rope_split(q, k, (cos, sin, True))
    except ValueError as e:
        assert "bf16" in str(e) or "fp16" in str(e), f"wrong error message: {e}"
        return
    assert False, "expected ValueError on fp32 dtype"


def test_public_export_from_sageattention():
    """Public API: `from sageattention import fused_rope_split` works."""
    import sageattention
    assert hasattr(sageattention, "fused_rope_split"), \
        "fused_rope_split not exported from sageattention package"


# --- Runner -------------------------------------------------------------

def _run_one(name, fn):
    try:
        result = fn()
        if isinstance(result, tuple) and result[0] == "SKIP":
            print(f"SKIP  {name}  ({result[1]})")
            return "skip"
        print(f"PASS  {name}")
        return "pass"
    except AssertionError as e:
        print(f"FAIL  {name}  -- {e}")
        return "fail"
    except Exception:
        print(f"ERR   {name}")
        traceback.print_exc()
        return "err"


def main():
    tests = [(n, fn) for n, fn in globals().items() if n.startswith("test_") and callable(fn)]
    results = {"pass": 0, "fail": 0, "skip": 0, "err": 0}
    for name, fn in tests:
        results[_run_one(name, fn)] += 1
    print()
    print(f"PASS {results['pass']}/{len(tests)}  "
          f"FAIL {results['fail']}  SKIP {results['skip']}  ERR {results['err']}")
    return 0 if (results["fail"] + results["err"]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
