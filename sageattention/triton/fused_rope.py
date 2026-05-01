"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Fused split-RoPE for Q and K. One Triton kernel pass that mirrors LTX's
`apply_split_rotary_emb` (comfy/ldm/lightricks/model.py:343) for both
tensors at once. Drop-in primitive a consumer can call before sage's
int8/fp8 quant+attention path; closes the structural kernel-side gap
versus per-block patches that fuse RoPE inline (e.g. KJNodes'
`LTX2MemoryEfficientSageAttentionPatch`). Clean-room implementation;
the math is the standard split-RoPE rotation:

    out_first  = first  * cos - second * sin
    out_second = second * cos + first  * sin

where `first = x[..., :D//2]` and `second = x[..., D//2:]` for the last
dim of x. Sage stays primitive: this is a standalone helper, not bolted
into the dispatcher.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_qk_split_kernel(
    Q_ptr, K_ptr, Cos_ptr, Sin_ptr,
    H, T, D,
    OUT_DTYPE: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    """One program per (t, h, b) -- 3-D grid matches sage's house style
    (`quant_per_block.py`, `quant_per_thread.py`). Reads D values from q
    + k (split as two halves of D//2), reads D//2 cos+sin, writes the
    rotated halves back in-place. q/k layout is contiguous
    `[B, T, H*D]`; cos/sin layout is contiguous `[B, H, T, D//2]` --
    matches LTX's split-pe convention."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    D_half = D // 2
    cols = tl.arange(0, BLOCK_HD)
    mask = cols < D_half

    qk_base = (b * T + t) * (H * D) + h * D
    cs_base = (b * H * T + h * T + t) * D_half

    q_first  = tl.load(Q_ptr + qk_base + cols,           mask=mask, other=0.0).to(tl.float32)
    q_second = tl.load(Q_ptr + qk_base + D_half + cols,  mask=mask, other=0.0).to(tl.float32)
    k_first  = tl.load(K_ptr + qk_base + cols,           mask=mask, other=0.0).to(tl.float32)
    k_second = tl.load(K_ptr + qk_base + D_half + cols,  mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(Cos_ptr + cs_base + cols, mask=mask, other=1.0).to(tl.float32)
    sin = tl.load(Sin_ptr + cs_base + cols, mask=mask, other=0.0).to(tl.float32)

    q_out_first  = q_first  * cos - q_second * sin
    q_out_second = q_second * cos + q_first  * sin
    k_out_first  = k_first  * cos - k_second * sin
    k_out_second = k_second * cos + k_first  * sin

    tl.store(Q_ptr + qk_base + cols,          q_out_first.to(OUT_DTYPE),  mask=mask)
    tl.store(Q_ptr + qk_base + D_half + cols, q_out_second.to(OUT_DTYPE), mask=mask)
    tl.store(K_ptr + qk_base + cols,          k_out_first.to(OUT_DTYPE),  mask=mask)
    tl.store(K_ptr + qk_base + D_half + cols, k_out_second.to(OUT_DTYPE), mask=mask)


def _torch_split_rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Pure-torch split RoPE matching `apply_split_rotary_emb`. Used as
    fallback when triton preconditions fail and as the test reference.

    x: [B, T, H, D] or [B, T, H*D]; cos, sin: [B, H, T, D//2].
    Returns same shape as x. Non-mutating."""
    if x.ndim == 3:
        B, T_x, HD = x.shape
        D_half = cos.shape[-1]
        D = D_half * 2
        H = HD // D
        x_4d = x.reshape(B, T_x, H, D).transpose(1, 2)  # -> [B, H, T, D]
        out = _torch_split_rope_reference(x_4d, cos, sin)
        return out.transpose(1, 2).reshape(B, T_x, HD)

    D = x.shape[-1]
    D_half = D // 2
    first  = x[..., :D_half]
    second = x[..., D_half:]
    out_first  = first  * cos - second * sin
    out_second = second * cos + first  * sin
    return torch.cat([out_first, out_second], dim=-1)


def fused_rope_split(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: tuple,
    *,
    use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply LTX-style split RoPE to q and k in one fused Triton pass.

    Parameters
    ----------
    q, k : torch.Tensor
        Shape [B, T, H*D] contiguous, dtype bf16 or fp16. Modified
        in-place when the triton fast path runs.
    freqs_cis : tuple
        (cos, sin) or (cos, sin, split_pe_bool). cos and sin must have
        shape [B, H, T, D//2]. split_pe_bool, when present, must be
        True; an interleaved-RoPE convention falls back to the torch
        reference (this primitive is split-only in v1).
    use_triton : bool, optional
        Set False to force the torch reference path (useful for
        accuracy comparison or if the triton kernel is suspected
        wrong). Defaults to True.

    Returns
    -------
    (q_rot, k_rot) : tuple[torch.Tensor, torch.Tensor]
        Same dtype and shape as inputs. The triton fast path
        modifies q and k in place (returned tensors are the same
        buffers); a non-contiguous input is silently copied to a
        fresh contiguous buffer before the kernel runs, so the
        caller's original tensor is not mutated in that case. The
        fallback path always returns fresh tensors. Either way,
        treat the inputs as consumed.

    Notes
    -----
    The fallback non-triton path matches LTX's
    `apply_split_rotary_emb` exactly. v1 supports only the split-pe
    RoPE variant (LTX 2.3 video + audio); interleaved variants and
    other model-class conventions fall back to torch silently.
    """
    cos = freqs_cis[0]
    sin = freqs_cis[1]
    split_pe = freqs_cis[2] if len(freqs_cis) > 2 else True

    def _fallback():
        return (
            _torch_split_rope_reference(q, cos, sin),
            _torch_split_rope_reference(k, cos, sin),
        )

    triton_eligible = (
        use_triton
        and q.is_cuda
        and split_pe
        and q.ndim == 3
        and cos.ndim == 4
    )
    if not triton_eligible:
        return _fallback()

    B, H, T, D_half = cos.shape
    D = D_half * 2
    if q.shape != (B, T, H * D) or k.shape != (B, T, H * D):
        return _fallback()
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"fused_rope_split requires q/k to be fp16 or bf16; got {q.dtype}"
        )
    if k.dtype != q.dtype:
        raise ValueError(
            f"q and k must share dtype; got {q.dtype} vs {k.dtype}"
        )
    # cos/sin contiguity is the kernel's documented precondition; LTX
    # always builds them contiguous, so an assert is cheaper than a
    # per-call .contiguous() copy that would silently reflow them.
    assert cos.is_contiguous() and sin.is_contiguous(), \
        "fused_rope_split requires contiguous cos/sin"

    q = q.contiguous()
    k = k.contiguous()
    out_dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    BLOCK_HD = triton.next_power_of_2(D_half)
    num_warps = min(max(BLOCK_HD // 32, 1), 8)
    _rope_qk_split_kernel[(T, H, B)](
        q, k, cos, sin,
        H, T, D,
        OUT_DTYPE=out_dtype,
        BLOCK_HD=BLOCK_HD,
        num_warps=num_warps,
    )
    return q, k


__all__ = ["fused_rope_split"]
