#!/usr/bin/env python3
# Last updated: 2026-04-25
"""Test the get_last_dispatched_kernel() telemetry helper.

This is the consumer-facing observability surface that lets a
downstream tracer record which kernel sageattn() actually dispatched
to, instead of mirroring sage's routing table or treating the kernel
as opaque.

Standalone script (no pytest); run via $VIRTUAL_ENV/bin/python so it
uses the editable install of sageattention.

Expected to be run on RTX 4090 / sm89 / CUDA >= 12.8. On other archs
some assertions about specific kernel names will need to change to
match what sage's dispatcher picks.
"""

from __future__ import annotations

import sys
import threading

import torch

import sageattention
from sageattention import (
    get_last_dispatched_kernel,
    sageattn,
    sageattn_qk_int8_pv_fp16_cuda,
    sageattn_qk_int8_pv_fp16_triton,
    sageattn_qk_int8_pv_fp8_cuda,
)
from sageattention.core import (
    KERNEL_FP8_CUDA_FP32,
    KERNEL_FP8_CUDA_PP,
    KERNEL_FP16_CUDA,
    KERNEL_FP16_TRITON,
    _record_dispatch,
    _reset_dispatch_for_test,
)


def _make_qkv(B=1, H=4, S=128, D=64, dtype=torch.bfloat16, device="cuda"):
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    return q, k, v


def test_initial_value_is_none():
    _reset_dispatch_for_test()
    assert get_last_dispatched_kernel() is None, (
        f"expected None before any dispatch, got {get_last_dispatched_kernel()!r}"
    )
    print("ok  initial value is None")


def test_helper_is_exported_from_package():
    assert hasattr(sageattention, "get_last_dispatched_kernel"), (
        "get_last_dispatched_kernel must be importable from the top-level package"
    )
    print("ok  helper exported from sageattention package")


def test_sageattn_dispatcher_records_fp8_pp_on_sm89():
    # On sm89 + CUDA >= 12.8 (the box this fork targets), sageattn()
    # routes unmasked calls to sageattn_qk_int8_pv_fp8_cuda with
    # pv_accum_dtype="fp32+fp16" -- aka fp8_cuda++ (SageAttention2++).
    _reset_dispatch_for_test()
    q, k, v = _make_qkv()
    _ = sageattn(q, k, v, is_causal=False)
    got = get_last_dispatched_kernel()
    assert got == KERNEL_FP8_CUDA_PP, (
        f"sageattn() unmasked on sm89/cuda12.8+ should record "
        f"{KERNEL_FP8_CUDA_PP!r}, got {got!r}"
    )
    print(f"ok  sageattn() dispatcher recorded {got!r}")


def test_direct_triton_call_records_fp16_triton():
    _reset_dispatch_for_test()
    q, k, v = _make_qkv()
    _ = sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=False)
    got = get_last_dispatched_kernel()
    assert got == KERNEL_FP16_TRITON, (
        f"sageattn_qk_int8_pv_fp16_triton should record "
        f"{KERNEL_FP16_TRITON!r}, got {got!r}"
    )
    print(f"ok  fp16_triton kernel recorded {got!r}")


def test_direct_fp16_cuda_call_records_fp16_cuda():
    _reset_dispatch_for_test()
    q, k, v = _make_qkv(D=128)  # fp16 cuda kernel needs head_dim 64 or 128
    _ = sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=False)  # default pv_accum="fp32"
    got = get_last_dispatched_kernel()
    assert got == KERNEL_FP16_CUDA, (
        f"sageattn_qk_int8_pv_fp16_cuda(pv_accum='fp32') should record "
        f"{KERNEL_FP16_CUDA!r}, got {got!r}"
    )
    print(f"ok  fp16_cuda kernel recorded {got!r}")


def test_fp8_cuda_variant_records_correct_subname():
    # The fp8 cuda kernel has multiple pv_accum variants. Each one
    # should record a distinct stable short name.
    _reset_dispatch_for_test()
    q, k, v = _make_qkv()
    _ = sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=False, pv_accum_dtype="fp32+fp32")
    got = get_last_dispatched_kernel()
    assert got == KERNEL_FP8_CUDA_FP32, (
        f"sageattn_qk_int8_pv_fp8_cuda(pv_accum='fp32+fp32') should record "
        f"{KERNEL_FP8_CUDA_FP32!r}, got {got!r}"
    )
    print(f"ok  fp8_cuda(fp32+fp32) variant recorded {got!r}")


def test_thread_isolation():
    # threading.local() means each thread sees only its own dispatch
    # value. A worker thread reading the helper before any dispatch
    # should see None even if the main thread has already dispatched.
    # Uses real kernel-name constants so the test stays inside the
    # KernelName Literal (no type-checker false flags).
    _reset_dispatch_for_test()
    _record_dispatch(KERNEL_FP16_TRITON)

    worker_observations: list[str | None] = []

    def worker():
        # Worker starts fresh: its thread-local has no `last` attr yet
        worker_observations.append(get_last_dispatched_kernel())  # expect None
        _record_dispatch(KERNEL_FP8_CUDA_PP)
        worker_observations.append(get_last_dispatched_kernel())  # expect KERNEL_FP8_CUDA_PP

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert worker_observations[0] is None, (
        f"worker thread should not see main thread's dispatch; "
        f"got {worker_observations[0]!r}"
    )
    assert worker_observations[1] == KERNEL_FP8_CUDA_PP, (
        f"worker thread should see its own dispatch; "
        f"got {worker_observations[1]!r}"
    )
    assert get_last_dispatched_kernel() == KERNEL_FP16_TRITON, (
        f"main thread's value should be untouched by worker; "
        f"got {get_last_dispatched_kernel()!r}"
    )
    print("ok  threads see isolated dispatch values")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; skipping kernel-dispatch tests.", file=sys.stderr)
        return 0

    test_helper_is_exported_from_package()
    test_initial_value_is_none()
    test_sageattn_dispatcher_records_fp8_pp_on_sm89()
    test_direct_triton_call_records_fp16_triton()
    test_direct_fp16_cuda_call_records_fp16_cuda()
    test_fp8_cuda_variant_records_correct_subname()
    test_thread_isolation()
    print()
    print("all dispatched-kernel telemetry tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
