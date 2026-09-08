# Bench discipline

Last updated: 2026-09-08

> **Two gated benches, one per model, and they gate different
> quantities.** `tests/test_sageattn_ltx_shapes.py` +
> `tests/regression_baselines.json` cover LTX 2.3 and Z-Image;
> `tests/test_sageattn_h3_shapes.py` + `tests/regression_baselines_h3.json`
> cover MiniMax H3 (promoted from a spike 2026-09-08, CHANGELOG v0.7.11).
> Both run gated from `tests/run_all.sh`. A green run on one says nothing
> about the other -- the models are architecturally different, not two
> sizes of the same thing.
>
> **The H3 gate deliberately does not gate accuracy.** Its baselines
> carry speed, peak VRAM and cross-kernel fidelity, and **no
> rtol-vs-SDPA entries at all**, so the shared gate skips that check
> because there is nothing to check rather than because a threshold was
> widened. On synthetic input an rtol against SDPA is not a measurement
> at H3 config, so gating on it would gate an artifact. H3 accuracy lives
> in `tests/spikes/spike_h3_real_activations.py`, on captured q/k/v.
> H3 correctness and VRAM also have their own surfaces
> (`tests/test_sageattn_consume.py` at fl2va S=41822,
> `tests/test_quant_offset_overflow.py`, `tests/test_short_seq_tail.py`,
> and the `tests/spikes/spike_h3_*.py` set).

L3 reference for CLAUDE.md. Load this when:
- a torch / triton / CUDA / sage-rev bump just happened
- you're about to change bench shapes or add a new shape baseline
- you suspect a perf number came from a different env than today's

## Env snapshot

Every wall-clock comparison in `test_sageattn_ltx_shapes.py` is pinned
to the version surface in `internal/bench_env_<date>.txt`. After any
torch / triton / CUDA / sage-rev bump, re-run the test and resnapshot.
Trigger doc + drift threshold: `CHANGELOG.md` / Recurring process
items / "Bench env re-snapshot."

## Cross-session comparison

Use the `torch_flash / sage_fp8++` ratio (today 2.66x at the primary
shape), not absolute time -- driver-thermal drift is 1-2% across cold
boots even with no code changes. If absolute fp8++ time drifts but
the ratio holds, it's the box, not the code. Full reasoning in
`docs/perf_research_framework.md`.

## Before changing bench shapes

Run `tests/bench_workload_profile.py` against a recent consumer
trace. Shape decisions made without checking the actual workload
distribution have drifted twice: the original
`self_attn_large_704x704x497` at seq=31776/d=64 was stale and never
matched production, and the same class of drift nearly recurred at
v0.4.1 until the workload-profile coverage check surfaced "every
load-bearing baseline MISS." The script is durable; the discipline
isn't free unless documented.

## The baselines JSON is the source of truth for shape names

(There are two now -- `regression_baselines.json` for LTX,
`regression_baselines_h3.json` for H3. The rule below applies to each.)

`check_regressions()` discovers anchors from the data, not from
hardcoded strings. The first regression-check landed with a hardcoded
`self_attn_large_704x704x497` shape-name lookup; when SHAPES renamed
in v0.4.1, the speedup-ratio gate went silently dead until
`/simplify` caught it. Pattern: any shape-aware logic in the bench
infrastructure must derive the shape set from the JSON, not hardcode
strings. `tests/test_regression_check.py` guards the dead-branch
class with a unit test (`test_speedup_line_appears`).
