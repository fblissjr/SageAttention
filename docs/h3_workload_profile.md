# MiniMax H3 workload profile

Last updated: 2026-09-08

The H3 counterpart to `docs/ltx_workload_profile.md`: where a DiT block spends
time and peak memory. Produced by `tests/bench_h3_block_profile.py`; re-run it
rather than quoting these numbers if the config or the shape changes.

## Measurement

One DiT block at **S=41,822** -- 1344x768, 124 frames, two keyframes, the
packed length the gated bench row uses and the shape the consumer's default
graphs render. Config read from source (`comfy/ldm/minimax/model.py:474-477`):
hidden 5376, 50 layers, 56 heads, head_dim 128 (inner 7168), ffn 14336.
Synthetic weights, bf16, sm89, sage at `auto` (which resolves to `fp8_cuda++`).

Peak is the transient each sub-module **adds** over the allocator's state on
entry, not the process peak -- that is the number that decides whether moving
a seam would help.

| sub-module | ms | % of block | peak transient MiB |
|---|---|---|---|
| norm (RMSNorm x2) | 2.00 | 0.6% | 858 |
| qkv_proj Linear | 59.59 | 19.1% | 1715 |
| **attention (sage fp8++)** | **111.08** | **35.6%** | 1433 |
| out_proj Linear | 19.77 | 6.3% | 429 |
| **mlp fc1 Linear** | 79.91 | 25.6% | **2287** |
| mlp fc2 Linear | 39.40 | 12.6% | 429 |
| block total | 311.75 | 100% | |

Both allocation figures reconcile exactly with the arithmetic, which is the
cheapest check that the instrument measured what it claims: fc1's output is
`S x 2*ffn` bf16 = 2287 MiB, the fused QKV buffer is `S x 3*inner` = 1715 MiB.

## Two findings, and both change a documented ranking

### 1. Attention is about a third of block compute, not almost all of it

35.6% here, against a render-level bound of **>= ~32%** derived independently
from an A/B (`docs/h3_attention_stack.md`). Two methods over different data
landing in the same place is the strongest evidence in this repo for what the
attention share actually is.

The claim it replaces -- "attention is almost all of it" -- was resting on 76%
measured at a sequence length past H3's legal ceiling. Ranking an attention
kernel bet should use roughly a third, so a 2x kernel win is worth about a
sixth of a render before decode is counted.

### 2. The MLP is bigger than attention, on both axes

**Time:** fc1 + fc2 is 38.2% of block compute against attention's 35.6%.

**Memory:** fc1's transient (2287 MiB) is the largest single allocation in the
block -- larger than the fused QKV buffer (1715) and larger than the attention
kernel's own working set (1433).

`self.fc1(x)` is evaluated before `linear_input_act` is called
(`comfy/ldm/minimax/model.py`), so that 2287 MiB materialises on **every**
path. The INT8 fusion in `comfy/ops.py::linear_input_act` avoids writing the
*post-SwiGLU* intermediate (a further `S x ffn` = 1143 MiB), not fc1's output.
So on a non-INT8 checkpoint the MLP's transient is larger still.

## What this does to this fork's ranking

**The FFN line was parked on a false premise.** `sage_ffn` and its forward work
were ranked at zero priority on the grounds that they are "LTX-motivated and
buy H3 little, since H3's time is almost entirely attention". H3's MLP is
*more* of the block than its attention is, on time and on memory both. That
does not make `sage_ffn` usable here as it stands -- it targets a GELU MLP and
H3's is SwiGLU with a `2*ffn` fc1, a different shape -- but the reason for
parking it was wrong, and a SwiGLU variant now has a real motivation on the
only model this fork targets.

**The memory levers are aimed off-target.** `sageattn_consume`, the caller-side
`v` clone and the `per_channel_fp8` transpose-buffer item all act inside the
attention path, which holds neither the largest transient nor the seam where
it is created. They recover hundreds of MiB downstream of a producer-side
approach that reportedly saves gigabytes, and the largest allocation in the
block is not in their path at all.

## Limits, so this is not over-read

**Isolation evidence, Cell A/B, not delivered.** One block, synthetic weights,
no sampler, no render. It cannot see L2 contention between neighbouring
modules, allocator state across a real step, or offload. This repo has been
burned precisely there: a primitive that benched 1.26-1.36x in isolation came
back +1.79% slower in-pipeline. **Read the shares, not the absolute times**,
and confirm anything load-bearing against a real render.

It also omits what sits outside a block and inside a render: adaln, rope, the
modulation adds, sampler overhead, text encoding, VAE decode, offload. Those
are in a render's denominator and push every share here down, which is the
direction that reconciles 35.6% of a block with >= 32% of a render.

Synthetic weights are legitimate for this question -- input distribution does
not change the work done or the allocation sizes -- but see
`docs/testing_practices.md` for where that stops being true.
