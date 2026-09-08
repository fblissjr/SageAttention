# MiniMax H3 workload profile

Last updated: 2026-09-08

The H3 counterpart to `docs/ltx_workload_profile.md`: where a DiT block spends
time and peak memory. Produced by `tests/bench_h3_block_profile.py`; re-run it
rather than quoting these numbers if the config or the shape changes.

## What was actually run

**No workflow, no sampler, no render.** One DiT block, synthetic weights,
sm89, sage at `auto` (resolving to `fp8_cuda++`). Config read from source
(`comfy/ldm/minimax/model.py:474-477`): hidden 5376, 50 layers, 56 heads,
head_dim 128 (inner 7168), ffn 14336.

**Two weight formats, and only one of them describes production.** The
consumer's H3 base is INT8 ConvRot, so every Linear in a real render goes
through `comfy/ops.py::linear_input_act` into `comfy_kitchen.int8_linear`,
and `fc2` folds the SwiGLU into its input quantizer. A first version of this
profile used plain bf16 `nn.Linear` and its numbers were wrong about
production by a wide margin -- see the correction below.

Two sequence lengths, because the split moves with S:

| | S | geometry |
|---|---|---|
| short | 41,822 | 1344x768, 124 frames (~5.2 s), fl2va, 2 keyframes |
| long | 104,030 | 1344x768, 345 frames (~14.4 s), t2v, the legal ceiling |

Peak is the transient each sub-module **adds** over the allocator state on
entry, not the process peak.

## Production path (INT8) -- these are the numbers to use

### S=41,822 (124 frames)

| sub-module | ms | % of block | peak transient MiB |
|---|---|---|---|
| norm (RMSNorm x2) | 2.00 | 1.0% | 858 |
| qkv_proj int8 | 29.77 | 15.0% | 1930 |
| **attention (sage fp8++)** | 110.58 | **55.7%** | 1433 |
| out_proj int8 | 6.20 | 3.1% | 715 |
| mlp fc1 int8 | 31.29 | 15.8% | **2502** |
| mlp fc2 int8+swiglu | 18.66 | 9.4% | 2144 |
| block total | 198.51 | 100% | |

### S=104,030 (345 frames)

| sub-module | ms | % of block | peak transient MiB |
|---|---|---|---|
| norm (RMSNorm x2) | 4.97 | 0.6% | 2134 |
| qkv_proj int8 | 73.87 | 8.3% | 4801 |
| **attention (sage fp8++)** | 675.57 | **75.6%** | 3563 |
| out_proj int8 | 15.41 | 1.7% | 1778 |
| mlp fc1 int8 | 77.35 | 8.7% | **6223** |
| mlp fc2 int8+swiglu | 46.65 | 5.2% | 5334 |
| block total | 893.83 | 100% | |

## Findings

Values live in the tables above; this section states directions and what
follows from them.

**Attention is the majority of block compute at both lengths, and its share
rises steeply with length**, because attention is O(S^2) where everything
else in the block is O(S). A single number for "H3's attention share" is
therefore not a well-formed claim -- quote it with a sequence length.

**The long-standing figure is vindicated, and my criticism of it was
half wrong.** I attacked it partly for being measured at an illegal length.
It was not: 362 frames is the top of H3's trained range, not an illegal length. The consumer withdrew the "345 is the largest legal count" claim on 2026-08-16 as an owner decision: 345 is the largest count *diffusers* emits, a fact about diffusers, while ComfyUI's node accepts far more and names ~124-362 as the trained range. This repo carried the withdrawn version for three weeks and propagated it on 2026-09-08. The long row above reproduces the figure
independently at a slightly shorter length on the same path, so the number
was sound. What survives as fair criticism is only that it was quoted as
though length-independent, and that the larger version circulating in
conversation was never the recorded one.

**The MLP never exceeds attention on the production path**, at either
length.

**The weight format changed two quantities by two different mechanisms, and
they need separate conditions.** An INT8 Linear being faster than a bf16 one
*raises attention's share of block time*. A fused epilogue removing the
SwiGLU concurrency *changes where the memory peak sits*. Neither result is
evidence for the other, and a reader who pairs them will reach a conclusion
neither supports.

**Memory is a separate question from time, and it answers differently.**
`mlp fc1` holds the largest single transient at both lengths -- above the
fused QKV buffer and above the attention kernel's working set. That survives
the weight-format correction below, because the terms involved scale
together and the ordering cannot flip.

## The correction, kept because it cost three wrong conclusions

The first version of this profile used bf16 `nn.Linear`. On that path
attention read 35.6% at 124 frames and the MLP 38.2%, and this document
concluded that **the MLP was the larger share and that `sage_ffn`'s parking
had been falsified**. It then concluded that the ranking *flipped* with clip
length. Both conclusions were artifacts of the wrong weight format:

| | bf16 (not production) | INT8 (production) |
|---|---|---|
| attention @ 124f | 35.6% | **55.7%** |
| MLP @ 124f | 38.2% | 25.2% |
| attention @ 345f | 57.1% | **75.6%** |
| block total @ 124f | 311.8 ms | 198.5 ms |

An INT8 linear is much faster than the bf16 one, so every Linear row was
inflated and attention's share understated. **`sage_ffn`'s parking was never
falsified.** The reasoning that parked it -- H3's time is overwhelmingly
attention -- holds on the path that ships.

The rule this violated is the repo's own: *measure the config that ships*. It
was violated in the file written to settle a ranking question, and the error
survived two rounds of correction because each round re-examined the shape and
never the weight format.

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
