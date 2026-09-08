# MiniMax H3 workload profile

Last updated: 2026-09-08

The H3 counterpart to `docs/ltx_workload_profile.md`: where a DiT block spends
time and peak memory. Produced by `tests/bench_h3_block_profile.py`; re-run it
rather than quoting these numbers if the config or the shape changes.

## What was actually run

**No workflow, no sampler, no render.** One DiT block, synthetic weights,
bf16, sm89, sage at `auto` (which resolves to `fp8_cuda++`). Config read from
source (`comfy/ldm/minimax/model.py:474-477`): hidden 5376, 50 layers, 56
heads, head_dim 128 (inner 7168), ffn 14336.

Two sequence lengths, because **the answer depends on the length and the
first version of this document did not say so**:

| | S | geometry |
|---|---|---|
| short | 41,822 | 1344x768, 124 frames (~5.2 s), fl2va with 2 keyframes |
| long | 104,030 | 1344x768, 345 frames (~14.4 s), t2v, the legal ceiling |

Peak is the transient each sub-module **adds** over the allocator's state on
entry, not the process peak -- that is the number that decides whether moving
a seam would help.

### S=41,822 (124 frames)

| sub-module | ms | % of block | peak transient MiB |
|---|---|---|---|
| norm (RMSNorm x2) | 2.00 | 0.6% | 858 |
| qkv_proj Linear | 59.59 | 19.1% | 1715 |
| attention (sage fp8++) | 111.08 | **35.6%** | 1433 |
| out_proj Linear | 19.77 | 6.3% | 429 |
| mlp fc1 Linear | 79.91 | 25.6% | **2287** |
| mlp fc2 Linear | 39.40 | 12.6% | 429 |
| block total | 311.75 | 100% | |

### S=104,030 (345 frames)

| sub-module | ms | % of block | peak transient MiB |
|---|---|---|---|
| norm (RMSNorm x2) | 4.98 | 0.4% | 2134 |
| qkv_proj Linear | 152.04 | 12.8% | 4267 |
| attention (sage fp8++) | 676.54 | **57.1%** | 3563 |
| out_proj Linear | 48.70 | 4.1% | 1067 |
| mlp fc1 Linear | 204.44 | 17.3% | **5689** |
| mlp fc2 Linear | 97.65 | 8.2% | 1067 |
| block total | 1184.34 | 100% | |

Allocation figures reconcile exactly with the arithmetic at both lengths,
which is the cheapest check that the instrument measured what it claims.

## The finding is a curve, not a number

**The attention/MLP ranking flips with clip length.**

| | attention | MLP (fc1+fc2) |
|---|---|---|
| 124 frames | 35.6% | **38.2%** |
| 345 frames | **57.1%** | 25.5% |

Attention is O(S^2) where the projections and the MLP are O(S), so a 2.49x
sequence length moves attention from a minority of block compute to a clear
majority. Both of the following are true, and neither is true unqualified:

- at a short clip, **the MLP is a larger share of the block than attention**
- at the ceiling clip, **attention is more than half the block**

**Correction, and it matters for how the earlier number was presented.** The
first version of this document paired the 35.6% figure with the render-level
bound of `>= ~32%` and called them two methods landing in the same place. They
are not measurements of one quantity: the bound was derived from renders at
345 frames, this profile's 35.6% is at 124 frames, and the quantity scales
with S. Consistent, yes; corroborating, no. That is the same axis-promotion
error this session has now made four times -- evidence and claim quantifying
over different things.

**What survives at both lengths: the memory result.** `mlp fc1`'s transient is
the largest single allocation in the block at 124 frames (2287 MiB, over the
fused QKV buffer's 1715) and at 345 frames (5689 MiB, over 4267). That
ordering does not flip, because both terms are O(S).

`self.fc1(x)` is evaluated before `linear_input_act` is called
(`comfy/ldm/minimax/model.py`), so that transient materialises on **every**
path. The INT8 fusion in `comfy/ops.py::linear_input_act` avoids writing the
*post-SwiGLU* intermediate (a further `S x ffn`), not fc1's output.

## What this does to this fork's ranking

**The FFN line was parked on a premise that is true at one clip length and
false at another.** `sage_ffn` was ranked at zero priority because H3's time
is "almost entirely attention". At the ceiling clip that is nearly right --
attention is 57% of the block. At a 124-frame clip it is wrong: the MLP is the
larger share. So the honest statement is that the FFN's value on H3 depends on
what people render, and nobody checked which clip lengths dominate actual use
before ranking it at zero. On memory it is the larger transient at both
lengths.

That does not make `sage_ffn` usable here as it stands -- it targets a GELU MLP and
H3's is SwiGLU with a `2*ffn` fc1, a different shape -- but the reason for
parking it was wrong, and a SwiGLU variant now has a real motivation on the
only model this fork targets.

**The memory levers are aimed off-target.** `sageattn_consume`, the caller-side
`v` clone and the `per_channel_fp8` transpose-buffer item all act inside the
attention path, which holds neither the largest transient nor the seam where
it is created. They recover hundreds of MiB downstream of a producer-side
approach that reportedly saves gigabytes, and the largest allocation in the
block is not in their path at all.

## VALIDITY WARNING: this measured a configuration nobody runs

**Added 2026-09-08, hours after the profile. Read before acting on any
number above.**

This profile used plain bf16 `nn.Linear` for the projections and the MLP.
**Production does not.** The consumer's H3 base is INT8 ConvRot, so every
Linear in a real render goes through `comfy/ops.py::linear_input_act` into
`comfy_kitchen.int8_linear` -- a different kernel with different speed and
different memory behaviour. Only the attention arm matches what ships.

Two consequences, in opposite directions, and neither is small:

**Time.** An INT8 linear should be faster than the bf16 one measured here.
If so, every Linear row above is inflated and **attention's share is
understated at both lengths** -- so the "MLP is larger at 124 frames"
finding may not survive on the real path at all.

**Memory.** A trajectory trace shows the block peaks not at `fc1` but just
after the SwiGLU, where fc1's output and the SwiGLU's output are live
together (5746 MiB at S=41,822, against 4320 reached during attention).
That concurrency is exactly what `linear_input_act` fuses away on the INT8
path -- the activation rides inside fc2's quantizer instead of writing a
full-size intermediate. **So the peak this profile found is plausibly an
artifact of the bf16 path and may not exist in production.**

This is the repo's own "measure the config that ships" rule, violated in
the file written to settle a ranking question. The shares here are
suggestive and the method is reusable; the numbers should not be quoted
about production until re-run against the quantized Linears, either by
constructing INT8 weights or by profiling a real render.

What survives regardless: attention's share rises steeply with clip length
(O(S^2) against O(S)), so any single-number claim about "the attention
share of H3" is wrong whatever the weight format.

## Limits, so this is not over-read

**Isolation evidence, Cell A/B, not delivered.** One block, synthetic weights,
no sampler, no render. It cannot see L2 contention between neighbouring
modules, allocator state across a real step, or offload. This repo has been
burned precisely there: a primitive that benched 1.26-1.36x in isolation came
back +1.79% slower in-pipeline. **Read the shares, not the absolute times**,
and confirm anything load-bearing against a real render.

It also omits what sits outside a block and inside a render: adaln, rope, the
modulation adds, sampler overhead, text encoding, VAE decode, offload. Those
are in a render's denominator and push every share here down. That direction
is why a block share and a render share are not interchangeable, and why the
two must not be quoted as confirming each other -- see the correction above.

Synthetic weights are legitimate for this question -- input distribution does
not change the work done or the allocation sizes -- but see
`docs/testing_practices.md` for where that stops being true.
