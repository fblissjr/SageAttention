# The H3 attention stack: what actually runs, and what reaches sage

Last updated: 2026-09-08

How this fork sits inside the shipped H3 graphs, why an attention win
multiplies against a share rather than a render, and the comparison
rules that follow. Extracted verbatim from `CLAUDE.md` on 2026-09-08.

**Read this before ranking any H3 attention work, and before comparing
this fork's accuracy against an approximate kernel's published figure.**
The referent rule at the bottom is the one that has caused the most
confusion.

## The attention share, and why it is weaker evidence than it looks

**Checked 2026-09-08. The only measured H3 attention share in this repo
is 76% of the step, and it was taken at S=109,126 -- a 362-frame packed
length, which is past H3's 15.0 s ceiling. 345 frames is the largest
legal count, so that measurement is at a shape nobody can render**
(`docs/minimax_h3_av_sampling.md`).

Three consequences, and the third is the one that matters:

1. **It is not 90%.** The figure repeated in conversation is higher than
   the figure on record. Quote 76%, with its S.
2. **It is at an illegal shape.** Not wrong as a kernel measurement --
   the entry is careful to say read it as a measurement at that S -- but
   it is not a statement about a render anyone performs.
3. **It is biased upward for real renders, by construction.** Attention
   is O(S^2) where most of the rest of a step is roughly O(S), so the
   attention share *rises* with sequence length. The measured share sits
   at an S well above the common case: a 124-frame fl2va render is
   S=41,822, under half that length. So the share at the shapes actually
   rendered should be **lower than 76%**, not higher. That direction is
   reasoning from the complexity, not a measurement -- nobody has
   profiled a legal H3 shape.

**What this does to ranking.** "Attention is almost all of it, so
attention work is the only work worth doing" is the argument that ranks
everything on this model, and it rests on one number, at an unrenderable
length, that is smaller than the version in circulation and points the
wrong way for real shapes. It may well survive a proper profile. It has
not had one.

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
  | **bottleneck** | mixed -- FFN is a real share (`docs/ltx_workload_profile.md`), attention is one part | attention dominates, but see the share note below -- "almost all of it" overstates what was measured |

  **The bottlenecks differ, so the work that pays differs -- but the H3
  half of this was wrong, corrected 2026-09-08.** The FFN line was
  written off here as "LTX-motivated and buys H3 little, because H3's
  time is in attention". A block profile says H3's MLP is a *larger*
  share than its attention, on time and on peak memory both
  (`docs/h3_workload_profile.md`). The shapes still differ -- GELU versus
  SwiGLU with a `2*ffn` fc1 -- so nothing existing drops in, but the
  motivation is real and was denied on a false premise. Conversely, attention-kernel
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
