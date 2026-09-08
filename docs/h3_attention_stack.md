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

**Never quote it without a sequence length.** Attention is O(S^2) where the
projections, the MLP and the norms are O(S), so its share of a DiT block
rises with clip length. A bare percentage for this model is not a
well-formed claim.

**Where the numbers live:** `docs/h3_workload_profile.md` carries the
per-sub-module table at two clip lengths on the path that ships, and
`tests/bench_h3_block_profile.py` regenerates it. Do not restate its values
here; this document goes stale, that one is dated and re-runnable.

**Status of the long-standing figure.** The share this repo quoted for
months was measured at the top of H3's trained range, and this repo called
it illegal for three weeks.

**The precise position.** At 24 fps 15.0 s is 360 frames exactly, and H3's 17n+5 grid straddles it: 345 lands at 14.375 s, the next value 362 at 15.083 s. So "362 exceeds 15.0 s" is arithmetically true and the original note was right about that. What does not follow is that 362 is unrenderable: nothing on the ComfyUI path enforces a 15.0 s limit -- its node accepts far more and names ~124-362 as the trained range -- and the consumer recorded on 2026-08-16 that the 345 bound is a fact about *diffusers*, which clamps there because 345 is the largest grid value at or under 15.0 s. So 362 is out of bounds under diffusers and in bounds under the path this fork targets, and a measurement taken there is at a demanding but renderable shape.

The profile reproduces the figure closely at a slightly shorter length on
the same path, so the number was sound. What survives as fair criticism is
only that it was quoted as though length-independent.

**Three denominators, and none of them are interchangeable.** The profile
reports shares of a **DiT block**. The consumer's bound is of **sampler
time**. A derivation done here from wall totals is of a **whole render**,
which additionally carries text encoding, VAE decode and model loading.
None of these confirms another, and they must never be paired as
corroboration -- this document did exactly that for part of 2026-09-08 and
it was wrong twice: once pairing a block share with a render bound, and
once treating a sampler-time floor and a wall-time floor as one result
because their figures were close.

**What this does to ranking.** The premise that ranks work on this model --
attention is where the time is -- holds on the path that ships, at every
length measured, and strengthens with clip length. Amdahl against a render
should use the render-level bound, not the block share.

## The render-level bound

The consumer's five-scene ladder renders identical geometry under stock
dense attention, this fork alone, and the approximate override alone. Let N
be non-attention time, constant across arms; N cannot exceed the fastest
arm's total, because attention time in that arm cannot be negative. That
yields a floor on attention's share for both the dense and the sage
configuration, the second being the one that ranks this fork's work. The
consumer's authoritative version is over **sampler time** and lives in their
`bench/results/` as a dated record with canvas, length and weight format
read off the source rather than written by hand; a version derived here from
wall totals is a different and larger denominator, and the two are not the
same quantity even though their figures are close.

Both are floors, not estimates: neither this fork nor the override makes
attention free. The derivation and its figures are recorded in CHANGELOG
v0.7.x for 2026-09-08; the consumer session derived the same bound from a
different field of the same record and landed in the same place.

Cite it with its limits: a bound from an A/B rather than a profile, one
seed, wall time including decode and load, one scene short of a full set on
the dense arm, and a split specific to that scheduler, shift, step count and
clip length.

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
  | **bottleneck** | mixed -- FFN is a real share (`docs/ltx_workload_profile.md`), attention is one part | attention is the majority at every length measured, rising with clip length (`docs/h3_workload_profile.md`) |

  **The bottlenecks differ, so the work that pays differs, and for H3 the
  profile confirms it.** The FFN line is LTX-motivated and buys H3
  little: on the shipped INT8 path attention is the majority of a DiT
  block at every length measured and the MLP never exceeds it -- shares
  and conditions in `docs/h3_workload_profile.md`. A bf16 profile
  briefly suggested the
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
  (none, on H3) and kernel errors.

  > **Dated record -- consumer-side e2e, 2026-08-14.** This is the only
  > home these figures have; nothing else in the repo records them, so
  > they are kept here as a record rather than pointed at.
  > Override-on against sage-alone at a 362-frame packed length -- the top
  > of the trained range, and a legal shape. An earlier version of this
  > block called it out of ceiling on a claim the consumer withdrew on
  > 2026-08-16; see `docs/h3_workload_profile.md`.
  > Override-on 493.4 s; sage-alone 794.7 s; ratio 1.61x.
  > The sage-alone arm ran `fp8_cuda++` while the graphs of the day
  > shipped `fp16_cuda`, so it understated.

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
