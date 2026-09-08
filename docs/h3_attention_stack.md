# The H3 attention stack: what actually runs, and what reaches sage

Last updated: 2026-09-08

How this fork sits inside the shipped H3 graphs, why an attention win
multiplies against a share rather than a render, and the comparison
rules that follow. Extracted verbatim from `CLAUDE.md` on 2026-09-08.

**Read this before ranking any H3 attention work, and before comparing
this fork's accuracy against an approximate kernel's published figure.**
The referent rule at the bottom is the one that has caused the most
confusion.

## The attention share: settled 2026-09-08, and it is a curve

**Quote it with a sequence length. A bare "attention is N% of H3" is not
a well-formed claim,** because attention is O(S^2) where the projections,
the MLP and the norms are O(S). Measured on the path that ships (INT8
Linears), one DiT block, `docs/h3_workload_profile.md`:

| clip | S | attention share of block |
|---|---|---|
| 124 frames, fl2va | 41,822 | **55.7%** |
| 345 frames (ceiling), t2v | 104,030 | **75.6%** |

**The long-standing 76% figure is vindicated, with one caveat retired and
one kept.** It was measured at S=109,126 -- a 362-frame length, past H3's
15.0 s ceiling, so at a shape nobody can render
(`docs/minimax_h3_av_sampling.md`). The profile above reproduces it
independently at S=104,030, which *is* renderable. So the number was
sound; what was wrong was quoting it as if it were length-independent,
and it is not 90%.

**Two denominators, and they are not interchangeable.** The shares above
are of a **DiT block**. A separate bound below puts attention at **>= ~32%
of a whole render**, which is a different and much larger denominator --
it includes sampler overhead, text encoding, VAE decode and offload. A
block share and a render share must never be quoted as confirming each
other; this document did exactly that for part of 2026-09-08 and it was
wrong.

**What this does to ranking.** The premise that ranks work on this model
-- attention is where the time is -- holds on the path that ships, at
both lengths, and strengthens with clip length. Amdahl against a *render*
should still use the ~32% floor rather than the block share.

**A bound now exists, from an A/B rather than a profile (2026-09-08).**
The consumer's five-scene ladder renders the same geometry under stock
dense attention, this fork alone, and the approximate override alone.
Let N be non-attention time, constant across arms; N cannot exceed the
fastest arm's total, because attention time in that arm cannot be
negative. Medians (wall total, 1344x768, 345 frames, 16 steps, one seed;
dense n=4 with one scene excluded for a cache hit, others n=5):

- **attention >= ~71% of a dense render**
- **attention >= ~32% of a render with this fork** -- the denominator
  that matters, because it is the configuration anyone actually runs

Both are floors, not estimates: neither this fork nor the override makes
attention free, so the true shares are higher. The consumer session
derived the same bound independently from per-scene sampler times and
got ~72% and ~34%; agreeing from two different fields is worth more than
either alone.

**So the premise survives, with a number attached, and the number is
about a third rather than "almost all".** Making attention free would cut
a real render by at most two thirds *in the dense configuration nobody
uses*, and by at most a third in the one they do. Amdahl on an attention
kernel improvement should use ~32%, and a 2x kernel win is then worth
around a sixth of the render, before VAE decode is even counted.

Cite it with its limits: a bound from an A/B, not a profile; one seed;
wall time including decode and load; and the dense-versus-approximate
split is specific to that scheduler, shift, step count and clip length.

**Superseded in part, 2026-09-08:** a block profile on the shipped INT8
path now gives 55.7% at S=41,822 and **75.6% at S=104,030**, the latter
independently reproducing the old 76% figure at a near-identical length.
So the old number was sound for near-ceiling clips; what was wrong was
quoting it as length-independent. See `docs/h3_workload_profile.md`.

**The gap:** LTX has `docs/ltx_workload_profile.md` -- sub-module shares
from a real render, the canonical input for ranking a perf bet. H3 has no
equivalent. That is the highest-value missing measurement on this model,
because it is upstream of every decision about what to optimise, and it
is a profiling run rather than a kernel day.

## Why H3 is not LTX, and why the benches split

Kept because it is how you read anything in this repo dated before
2026-08-04, and because the bottleneck difference is what decides which
work pays. LTX is parked as of 2026-09-08; this stays for interpreting
the older record.

- **Label the model on every measurement, claim, and doc.** LTX 2.3 and
  MiniMax H3 are architecturally different, not two sizes of the same
  thing, so a number from one is not evidence about the other:

  | | LTX 2.3 | MiniMax H3 |
  |---|---|---|
  | in this repo since | ~2026-04 | **2026-08-04** (`3f3a121`) |
  | attention sites | self-attn + **masked cross-attn** (headline shape) | **one** call site, no cross-attn |
  | mask | load-bearing (drove the v0.5.5 kernel) | `mask=None` hardcoded; unreachable |
  | sequence | separate q/kv streams | one packed `[text\|refs\|audio\|video]` |
  | **bottleneck** | mixed -- FFN is a real share (`docs/ltx_workload_profile.md`), attention is one part | **attention: 56% of a DiT block at 124 frames, 76% at the ceiling** (INT8 path). Quote with an S |

  **The bottlenecks differ, so the work that pays differs, and for H3 the
  profile confirms it.** The FFN line is LTX-motivated and buys H3
  little: on the shipped INT8 path attention is 56% of a DiT block at
  124 frames and 76% at the ceiling, and the MLP never exceeds it
  (`docs/h3_workload_profile.md`). A bf16 profile briefly suggested the
  MLP was larger; that was an artifact of measuring a weight format
  nobody runs, and this paragraph carried the wrong version for part of
  2026-09-08. Conversely, attention-kernel
  quality and speed is nearly the whole lever on H3 and only a fraction of
  one on LTX. Rank any perf bet against the model it targets, not against
  the repo in general; the Amdahl ceiling is different per model and a
  wedge that is real on one can be noise on the other.

  Consequences that have already bitten: the v0.5.5 native-mask kernel is
  LTX-motivated and buys H3 nothing; `fp16_cuda`'s silent mask-drop
  disqualifies it for LTX but not for H3. Anything dated before 2026-08-04
  is LTX/Z-Image by construction -- cite it as corroborating a pattern,
  never as confirming an H3 result.

  **Each model has its own gated bench, and they gate different
  quantities.** `tests/test_sageattn_ltx_shapes.py` +
  `tests/regression_baselines.json` for LTX;
  `tests/test_sageattn_h3_shapes.py` + `tests/regression_baselines_h3.json`
  for H3 (promoted from a spike in v0.7.10). Both run gated in
  `tests/run_all.sh`. The H3 file gates **speed, peak VRAM and cross-kernel
  fidelity only** -- its baselines carry no rtol-vs-SDPA entries at all, so
  the shared gate skips that check by construction rather than by a loose
  threshold. That is the synthetic-input rule under Testing applied to a
  gate: an rtol against SDPA on `torch.randn` is not a measurement at H3, so
  gating on it would gate an artifact. H3 accuracy stays with
  `tests/spikes/spike_h3_real_activations.py` and its captured q/k/v. H3
  sequence lengths are derived in-file from the consumer node's own geometry
  rules, not hand-copied, so a node-side geometry change shows up as a shape
  change rather than as silent drift.

  **On H3, most attention no longer reaches sage.** Since 2026-08-14 the
  shipped consumer graphs chain a third-party block-sparse-attention CUDA
  override (Sol-Attn, arXiv 2607.24027) above sage and give it H3's single
  full-packed-length DiT attention call; sage is the fallback link and
  receives only what the override declines -- depth-gated dense blocks,
  steps outside the sigma window, sub-`min_tokens` calls, masked calls
  (none, on H3) and kernel errors. Consumer-side e2e at 362 frames
  (an out-of-ceiling length -- H3 rejects past 15.0 s after the 17n+5
  snap, so 345 is the largest legal count; the ratio is still a ratio,
  but it was taken at a shape nobody can render),
  2026-08-14: 493.4 s against 794.7 s sage-alone (1.61x); that baseline ran
  `fp8_cuda++` while the graphs of the day shipped `fp16_cuda`, so it
  understated.

  **That last clause has since expired -- re-checked 2026-09-08.** Every
  API graph in the consumer repo now sets the sage node's mode to `auto`
  (92 of 92), and `auto` on sm89 resolves to `fp8_cuda++`. So the
  baseline and the shipped graphs run the *same* kernel today and the
  "understates" correction no longer applies to a current render. Keep
  the original reading only for the dated 2026-08-14 measurement it
  describes. Consequence worth carrying: any advice that starts "you are
  probably on fp16, switch to..." is advice about a configuration nobody
  runs -- and the real-activation accuracy figures under Testing are the
  ones that bear on the mode question, not the synthetic table.

  **Sage is not idle in the override-on arm, and an earlier version of this
  block said it was.** The "zero DiT calls" figure was read off the
  `min_tokens` gate alone and ignored the sigma window. The override's
  compose gate applies both, and a call failing *either* falls through to
  the previously installed patch -- ours -- so every step outside the sigma
  window runs the whole DiT on sage.

  That mechanism is read from the override's `_compose_module_patch` and is
  the part to rely on. The *share* is not: "5 of 16 at `0.2 / 0.9`" is the
  consumer's computation, taken on trust here and **not independently
  verified** -- it needs the sigma schedule under that scheduler and shift,
  which is not derivable from our source. It moves with step count,
  scheduler, shift and window. **Read it out of `get_dispatch_counts()`
  rather than quoting a number** -- the count is the only thing separating
  "sage handled a share" from "sage was bypassed", and both look identical
  in a log.

  So an H3 attention-kernel win multiplies against the dense share rather
  than against the render, and the quant-offset ceilings (CHANGELOG v0.7.0
  int32 fix; the `csrc/fused` uint32 ceiling under Known kernel bugs) bind
  on that path only. The sparse kernels are a separate implementation --
  `int64_t` strides and `size_t` offsets, read 2026-08-14 -- so they do
  not share that defect.

  **H3's conditioning region is multi-modal; do not reason about it as
  "the audio sink".** The packed sequence is
  `[text | refs | audio | video]` and `refs` is image *or video*
  references, which in a reference-heavy graph are the largest part of
  the conditioning region by a wide margin -- far larger than audio. The
  override's `sink_conditioning` knob is documented in terms of keeping
  generated audio intact, and that is one modality of a packed
  audio-video output, not the whole of what conditioning carries. When
  the override narrows which conditioning rows keep dense queries, the
  rows it makes sparse are mostly reference rows, and a *video*
  reference is temporally structured in a way an image reference is not.
  Any claim about that knob taken on a t2v graph (no refs, so the region
  is text+audio and text is under one 64-row block) generalises to
  nothing.

  **Never compare an approximate kernel's accuracy number to ours without
  checking what its reference computes.** A block-sparse kernel's
  correctness bench typically grades it against an eager implementation of
  *its own algorithm at the same settings*, so the approximation is on both
  sides and cancels: that number is implementation fidelity and contains no
  approximation error at all. Ours is distance from exact attention. The
  two are not the same quantity and no metric conversion reconciles them --
  check the referent before the metric. Sol's harness makes both available
  (fidelity ~0.9999, and separately its distance from its own dense limit);
  only the second is comparable to a sage rtol, and **both of its published
  figures are taken at `T=512, H=4` on `torch.randn`**, which is a
  degenerate regime for block routing (8 blocks of 64) on structureless
  input. Expect any approximation-vs-dense figure taken there to be
  pessimistic by a wide margin, for the same reason our own synthetic
  numbers are -- see the real-activation correction under Testing.
