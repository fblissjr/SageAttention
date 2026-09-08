# Drift audit and forward directions

Last updated: 2026-09-08

A dated audit, not a plan. `CHANGELOG.md` remains the single source of
truth for open triggers (Backlog) and closed decisions (Decision log);
`docs/roadmap.md` remains the tiered forward record. This file exists
because a single session surfaced a *pattern* those two are shaped wrong
to hold: a list of claims that were true when written, went false
without anything failing, and were only found by someone happening to
look. The pattern is the finding. Individual items below are cross-
referenced to their permanent homes and should not be maintained here.

**Do not treat this as a work queue.** Where an item has a trigger it is
recorded in the Backlog. Where it is closed it is in the Decision log.
What is here that is nowhere else: the method, the negatives, and the
argument for one specific piece of new machinery.

## The thesis

This repo's recurring defect is not wrong kernels. It is **instruments
and documents that cannot fail** -- checks that pass because the
condition they guard was never reproduced, and prose that stays green
because nothing ever evaluates it. `CLAUDE.md` already carries four
worked instances of the first under Testing. This audit is about the
second, and it is worse, because a stale document is indistinguishable
from a current one at the moment you read it.

Every item in Findings A shares one shape:

> A true statement was written down with a date. The thing it described
> then changed. Nothing in the repo evaluated the statement again, so
> nothing turned red. It was found by a person reading, months later,
> for an unrelated reason.

The cost is not the wrong sentence. It is that decisions get made on it.

## Method

What produced these findings, so it can be repeated:

1. **Re-read the source, not the note.** Every claim about an upstream
   file was checked by opening that file today, not by trusting the
   `path:line` in our docs. Two of the claims had drifted; one of the
   pointers no longer resolved to what it named.
2. **Grep the docs for their own falsifiers.** Sweeping committed
   markdown for the specific strings a change would invalidate
   (a toolkit version, a flag name, "no X exists") found stale claims
   faster than reading the docs would.
3. **Ask what state would turn a green check red, then create it.** The
   VRAM gate branch was mutation-tested by disabling it; the new unit
   test was mutation-tested the same way. A test that has only ever been
   green is not known to be a test.
4. **Check the caller you are about to accuse.** See A1: the first
   version of that finding was wrong in a way that would have sent
   someone to "fix" correct code.
5. **A mutation can mask itself.** Found while proving a new check could
   fail: the check compares a value against the checkout's own VCS
   state, and the obvious way to break it -- editing the source file --
   makes a tracked file modified, which moves *both* sides of the
   comparison and turns the mutation invisible. It passed, and passing
   there means nothing either way. The fix is to make the mutation
   without dirtying the tree: a detached worktree, the mutation
   committed inside it, then the condition staged there. Generalises to
   anything that reads the state its own edit changes -- if the mutation
   perturbs the oracle, a green result is uninformative rather than
   reassuring.

## Findings A: claims that had gone false

Each is fixed; the entry records what it cost and what would have caught
it earlier.

**A1. A documented caller-side memory optimisation had lost its
precondition.** The `sageattn_consume` / `sageattn_consume_prefers_cloned_v`
pair is built around a caller that gives `v` its own storage. Upstream
now hands three views of one fused QKV buffer, wrapped in a container
that stores without copying, so on that path the entry point is correct,
bit-identical, and saving nothing. Home: Decision log.

*This one also produced the audit's own worst error.* The first draft
generalised from "ComfyUI's built-in path no longer clones" to "no
caller clones", which is false -- the consumer attention-patch node
owns the H3 call site and does clone, gated on the predicate, with a
wiring test of its own. The accurate finding is narrower and more
useful: **the two paths now differ in per-call memory behaviour on the
same model**, so which one handles attention decides whether the
headroom exists. The error is the asymmetric-verification habit already
noted in this repo's memory: the caller being disputed got verified, the
caller being agreed with did not.

**A2. A `path:line` pointer no longer resolved.** The reference to H3's
single attention call site had drifted by nearly thirty lines. This
repo's own rule is that pointers must resolve, and there is nothing that
checks them.

**A3. The bench env snapshot silently omitted sage itself.** An editable
install lists as `-e file:///...`, and the snapshot's `grep` was anchored
on package names, so every snapshot ever taken pinned torch and triton
and stayed silent on the sage build behind the numbers. Home: v0.7.x.

**A4. Every bench log recorded `sage: ?`.** The header read a
`__version__` the package never defined. Same hole as A3, in the file
whose entire job is provenance. Now reports version plus source revision
plus a dirty marker. Home: v0.7.13.

**A5. The one-shot runner had been unconditionally red for about four
months.** A baseline row predated a routing change that moved it to a
different kernel. Because the runner aborts on that gate's exit code,
every step after it -- a second bench, the correctness suites, a spike
-- had not run from the runner in all that time. A permanently red gate
is worse than no gate: it cannot distinguish a new regression from the
known one, and it silently truncates everything downstream. Home:
v0.7.12.

**A6. A gate branch shipped with no unit test, in a file that has one
per branch.** Added the same day as the branch. Home: v0.7.13.

**A7. The build's `clean` broke sibling environments.** Extension
modules are tagged per interpreter and live in the source tree every
editable install points at, so a blanket wipe served one environment and
took the others out, with no error at the moment of breakage. Home:
v0.7.11.

**A8. A guard outlived its cause by a release.** A toolkit was
blocklisted for miscompiling the framework headers of the day. Those
headers were replaced; the miscompile stopped reproducing. The toolkit
was never fixed -- the code it choked on stopped existing. The finding
recorded a date but no *expiry condition*, so nothing prompted a
re-check. Home: v0.7.9.

## Findings B: checked and sound

Negatives are load-bearing here: they are the difference between "we
believe this is fine" and "this was checked on a date, by a method".

**B1. The framework's new fused-backend path for rank-3 inputs does not
reach us.** The release notes are explicit that it may alter numerics,
which would silently change our reference comparand. Every tensor we
hand the reference is rank-4, in this repo and in the consumer's bench
scripts, and the consumer's grading harness additionally pins the math
backend. Not a risk here; would be for any caller that flattens batch.

**B2. Grouped-query attention support added to the memory-efficient
backend does not apply.** Both tracked models are multi-head, not
grouped.

**B3. No deprecation warnings** on import or on a dispatched call under
`-W error::DeprecationWarning`.

**B4. The framework's own custom-op checker cannot grade our ops, and
its failure mimics a real defect.** Proven with a control that passes on
one dtype and fails identically on the other. The sub-tests that do run
pass, including the one that exercises the fake-kernel registration.
Home: Decision log.

**B5. The consumer's use of our surface is correct.** It reads dispatch
counts rather than inferring coverage, feature-detects the clone
predicate via `getattr` with a fallback for older forks, and does not
copy our internal arch list.

**B6. Both tracked environments import and run.** Including a newer
interpreter, where every third-party node loaded cleanly.

## Recommendations

Ranked by how much silent failure they remove per unit of work.

**R1. Add an upstream-contract test. DONE 2026-09-08 --
`tests/test_upstream_contracts.py`.**

Four checks, in `run_all.sh`, parsing the consumer source rather than
importing it: H3's attention call site passes no mask; the single-owner
container protocol still has `peek`/`take`; whether the caller gives `v`
independent storage (informational); whether the consumer still
introspects a signature around `attn_mask`. Each names the documented
claim it guards, so a failure sends the reader to the sentence rather
than to a debugger.

Verified against a fabricated tree where every claim is false: the two
load-bearing checks fail with actionable messages, and a missing
consumer skips reporting "ran 0 of 4 checks. This is a SKIP, not a
pass." The original design constraints are met and the reasoning for
them is below, kept because the next check added here should follow it.

The original argument:

**R1. Add an upstream-contract test. This is the one that matters.**

Findings A1 and A2 are the same bug: we depend on specific upstream
behaviour, we wrote it down, and nothing evaluates it. A1 changed a
memory characteristic; A2 changed only a line number, but the next one
might change which kernel gets reached.

Proposal: a test that asserts, against the *installed* consumer stack,
the handful of upstream facts this fork's design depends on. Candidates,
all currently documented as prose in `CLAUDE.md`:

- the tracked packed-sequence model's attention call site passes no mask
- q/k/v arrive wrapped in the single-owner container protocol
- whether the caller gives `v` independent storage (informational, and
  the exact fact that drifted)
- the consumer gates masked calls on `attn_mask` being a *named*
  parameter of the dispatcher -- already enforced from our side in
  `tests/test_dispatched_kernel_telemetry.py`, but not from theirs

Design constraints, because this is easy to build wrong:

- **Skip loudly, never silently.** The consumer stack is not a
  dependency of this fork. If it is not importable the test must report
  "skipped: consumer not installed", not pass. A skip that reads as a
  pass reproduces the exact failure class this is meant to end.
- **Assert the fact, not the line number.** Parse or import; do not
  match `path:line`. A test that breaks on reformatting will be deleted.
- **Informational rows are allowed and should be printed, not
  asserted.** The clone is a good example: its absence is not a defect,
  it is a change in what our entry point buys. The test should say so
  and stay green.
- **Each assertion names the doc claim it guards**, so a failure sends
  the reader to the sentence that needs editing.

**R2. Every cross-repo claim gets a re-check condition, not just a
date.** A8's guard survived its cause because the note said when it was
established and not what would make it stale. `docs/moving_targets.md`
already argues this; the gap is that nothing enforces it. Cheapest
enforcement is convention plus review: a claim about a fast-moving
dependency is incomplete without the sentence "this stops being true
when X".

**R3. A gate is never left red.** A5 cost four months of a truncated
suite. Either the baseline is updated with a named, dated cause, or the
gate is marked expected-red in a way the runner understands and reports.
"We know about those two lines" is not a mechanism.

**R4. Prefer deleting a primitive to maintaining a redundant one.** See
D1: at least one shipped primitive is now done better, in-place, by a
library the consumer already loads.

## Does any of this explain unsatisfying renders?

**Answered 2026-09-08, from evidence that already existed.** The
consumer repo had already run a blind, seed-matched, five-scene ladder
on 2026-09-03 pitting stock dense attention against this fork alone and
against several stacked configurations, and graded the pairs. It had not
been read in this direction. The result:

**Dense versus this fork alone: indistinguishable on four of five
scenes** -- graded "same" or "can't tell", with both halves tagged good.
On the fifth the grader gave dense a narrow win and wrote that both were
"pretty good given all the motion". That fifth scene is the one where
*every* arm loses to dense, including the two approximate-attention
configurations and the step-distilled one, which makes it a
scene-difficulty result rather than a finding about any kernel.

Two things make this stronger than a five-sample result usually is.
First, **every arm above dense carries this fork**, so a defect here
would have to show up in all of them rather than in one isolated pair.
Second, dense-versus-ours is the only contrast in the set that isolates
this fork, and it is the one that came back clean.

**Where dissatisfaction more plausibly comes from, on the same data:**
the step-distilled arm lost to dense on **all five** scenes. That arm is
confounded by construction -- the repo's own note records that it
carries this fork, the approximate override, and a distillation-specific
override setting together, so no arm renders it in isolation and the
loss does not attribute to any one of them. But it does say the
attention kernel is not the first place to look.

**Caveats, because this is a real result and should not be overstated.**
One seed, one pair per contest, five pairs total -- the verdict file's
own reading field says it is "a preference over distributions", and with
one pair the distribution is a single sample. The per-clip tab of the
same session was never scored, only the pairwise one, and the export is
marked partial. Enough to say there is no gross defect; not enough to
rule out a small quality cost.

**And the build under test was not this one -- but the source was.**
Two different axes, and it is worth keeping them apart because a result
on one gets quoted as covering the other.

*Source axis, verified:* no attention kernel source changed between
those renders and now. `git log` over `csrc/` and `sageattention/` since
that date returns only a version export and the deletion of a helper
that was never on the attention path. No `.cu`, no `.cuh`, no dispatch
or quantisation module.

*Toolchain axis, not verifiable:* the same source is now compiled under a
newer language standard against newer framework headers. The toolkit
comparison run the same day showed bit-identical output between two
compilers, and that does **not** reach this -- it holds one standard and
one set of headers fixed while varying the compiler. The comparison that
would reach it cannot be run, because the older standard will not
compile against the current framework at all.

So the residual is narrow and stated exactly: identical kernel source,
recompiled under a changed standard and changed headers, with the
compiled result unverified against the build the grading used. That is a
much smaller gap than "the evidence is about an old build", and it is
still a gap.

*A note on the corroboration.* The consumer-side session read the same
verdict file and reported the same five results. That is worth
something, since the verdicts are verbatim from the record either way --
but it was **not** two blind reads. Each session had told the other what
it was looking for before the other looked, so the framing travelled in
at least one direction and probably both. Agreement between two readers
who have exchanged notes is weaker evidence than it feels like, which is
the same trap as the rest of this document.

**The one thread worth pulling, and not yet.** The single loss came with
a named artifact on the highest-motion scene. On one pair that is not
separable from sampling. Settling it is cheap -- that scene, a couple
more seeds, same protocol, the two isolating graphs -- and it is a
render budget decision for the owner, not something to launch off a
single grader note.

**What this does not retire** is the gap below: there is still no
standing accuracy gate for this model, so the next regression would
again be found by someone looking rather than by something failing.

The original analysis follows, and its conclusion is unchanged.

**Nothing found here changes kernel output.** The toolkit change was
verified bit-identical across both fp8 accumulator variants, the
alternate kernel and the masked path. The language-standard bump is a
build flag. The clone drift (A1) is a memory characteristic; the entry
point is bit-identical with and without it. So none of it is a
numerical cause, and proposing one from this list would be
story-telling.

**What this audit does explain is why nobody would have found out.**
Three structural gaps, two of which were closed today and one of which
is by design:

1. Until this session the gated bench had **no coverage of the packed
   audio-video model at all**. Closed.
2. The one-shot runner had been red for months, so every step after the
   first gate -- including the correctness suites -- had not run from it
   (A5). Closed.
3. **There is still no accuracy gate for that model, and that is
   deliberate.** A synthetic-input distance from the reference is not a
   measurement at its config; the numbers it produces are dominated by
   cancellation and are roughly four times worse than reality. So the
   new gate covers speed, memory and cross-kernel fidelity only, and
   accuracy lives in a spike that needs captured activations and is run
   by hand.

The consequence is worth stating plainly: **if this fork's output
quality on that model regressed, nothing here would say so.** That is a
real gap, and it is the strongest argument in this document for the
capture-based harness being run on a schedule rather than on suspicion.

**Diagnostic order, cheapest first.** These are ordered to separate
attribution before tuning anything, because the first question is not
"which setting" but "whose output is this".

1. **Establish that these kernels ran at all.** On this model the
   shipped graphs chain an approximate-attention override above us; it
   takes the full-length calls and we receive only what it declines. A
   render can be mostly not-ours and look identical in a log. Read the
   dispatch counters on a real render. Until that number exists,
   attributing quality to this fork is unfounded -- and this repo
   already has a note about a share being taken on trust rather than
   derived.
2. **Confirm which mode the graph actually uses.** The two arms differ
   on real activations, and the difference is much smaller than the
   synthetic bench implies -- but it is not zero, and it is the one
   lever with a real-activation number behind it. See the Testing
   section of `CLAUDE.md` for the figures and, importantly, for which
   of them are synthetic and must not be quoted.
3. **Separate the two phenomena that get merged.**
   `docs/sparse_attention_quality_gating.md` exists for exactly this:
   broad decay near the top of the trained frame range, whose lever is
   frame count, versus threshold-graded content instability, which is
   the approximation's own artifact. They look similar and have
   different fixes. That doc also specifies the gate procedure --
   intra-clip, video not stills, and prove the knob fired.
4. **Only then, measure this fork's accuracy properly**, with the
   capture-based spike on q/k/v from a real forward. It is the only
   instrument here whose number means anything for this model.

**One candidate ruled out.** The known `uint32` offset ceiling in the
quantisation pre-kernels is documented as unreachable on this class of
card: the sequence length required exceeds what the memory budget
allows, and the largest reachable render measured well inside the
limit. It is a latent ceiling with a stated trigger, not a live defect.


## Where to dig further

**D1. Overlap with the consumer's kernel library. PARTLY DONE
2026-09-08 -- the first item is retired.** ComfyUI routes rotary
embedding through `comfy_kitchen` on **both** tracked models: the LTX
path via `apply_rope_split_half`, the packed audio-video path via
`rms_rope_split_half_`, the latter fused with RMSNorm and applied in
place on the QKV buffer at the same call site that then invokes
attention. That is strictly more than this fork's `fused_rope_split`
did, and nothing imported ours. It shipped on a "structural kernel-side
gap" claim the consumer retracted after measuring the share, and was
kept on the argument that a future DiT consumer might adopt it -- an
argument the library's presence in core has now closed. Removed in
v0.7.14 under the Backlog trigger that already existed for it.

**The rest of that capability set, enumerated 2026-09-08 -- and the
finding is bigger than a helper.** The library exposes 65 public
callables. Most do not overlap us: its quantisation entry points are
rowwise/tensorwise for linear layers where ours are the per-warp and
per-block granularities our own QK kernel's layout requires; its int4 and
mxfp8/nvfp4 paths are for weight formats we do not handle; its fp8
storage helpers quantize, where ours *reads* the four conventions a
consumer already stores. None of those is duplication.

**But it ships INT8 attention, and it runs on our arch.**
`int8_attention` computes inference SDPA with signed INT8 Q/K/V and
unsigned INT8 P, in the same `[batch, heads, sequence, head_dim]` layout
we take, with mask support, grouped-query support, and head dims padded
to 64/128/256 tiles. `int8_attention_is_available()` returns True on this
box. It also ships `prequantize_int8_attention` /
`int8_attention_from_prequantized`, which is the same idea as our
`sageattn_consume`: quantize without allocating the output, do not retain
the float inputs so the caller can free them, preserve stream ordering.

That is not a helper overlapping a helper. That is the fork's core
function, and the consume pattern we built, available from a library the
consumer already loads.

**Two differences that mean this is a question and not a verdict.** Ours
is INT8 QK with **fp8** PV; theirs is INT8 throughout, including P and V.
Different precision profile, and our own record is explicit that 8-bit
versus 16-bit PV is where the accuracy difference lives -- so int8 PV is
a third point on that axis and nobody has measured it. And theirs applies
a fused block-Hadamard rotation to Q and K before quantizing, which is a
stronger outlier treatment than our `smooth_k` and a technique this fork
does not implement at all.

**Status: unmeasured, deliberately.** A first speed and VRAM comparison
was attempted and discarded -- the GPU was already at 93% utilisation
running someone else's capture bench, and this fork's own number came
back roughly twice its known value, so both arms were contaminated. The
comparison needs an idle card. When it runs: speed and peak VRAM are
valid on synthetic input, and **accuracy is not** -- both implementations
are approximations of exact attention, so a `randn` comparison is
misleading in the pessimistic direction for both, and the real-activation
harness is the only instrument that answers it.

**Why this matters more than the rope helper did.** If that kernel
matches or beats ours at H3 shapes on speed and holds up on real
activations, the honest conclusion is that this fork's attention work has
been overtaken on its own ground, and the question becomes what is left
that is ours. If it does not, we learn what our fp8 PV path is actually
worth against a serious comparand, which is a number this repo has never
had. Either outcome is worth more than most of the queued work.

**D2. That library is also the binding model worth copying.** It carries
no framework symbols in its dynamic dependencies and registers its ops
on the Python side, so it survives framework upgrades without a rebuild.
This fork spent a session on a rebuild forced by a header change. This
is a rewrite of the binding layer, not a tweak, and what it buys out is
a short rebuild -- so the honest case for it is portability to machines
that cannot compile, not local convenience. Wanted before committing:
the per-call cost of that boundary at our shapes, since our kernels fire
thousands of times per render.

**D3. The consumer's install of that library is a supply-chain edge we
have already been bitten by.** It is pinned to an exact version, and a
locally built variant carries a local version segment that the pin still
matches -- so a routine requirements reinstall silently replaces a
purpose-built kernel with the stock wheel. It happened during this
session's environment work. The consumer's launcher already prints a
mismatch warning; the dig is whether anything *fails* on mismatch, or
whether a warning at startup is enough for a build that changes kernel
behaviour.

**D4. Attribution on the packed audio-video model: sequencing agreed
with the consumer-side session, 2026-09-08.** The original framing here
was "measure the override's dense share". That is the wrong first move
and the consumer session corrected it, so the corrected version:

*Ask what this fork produces before asking how much of the render was
this fork.* There is a settled bench graph that wires our attention node
and no approximate-attention override -- the sage-alone arm, and the one
the override work was measured against. Rendering that answers the
question directly: if its output is fine, the problem is upstream of us
and the share is irrelevant; if it is not, it is ours and the share is
still irrelevant. The share measurement explains a result; it does not
produce one.

*Do not start on the shipped graph.* Its composition is
length-dependent: the override declines every call below a token
threshold, and a short clip can sit just under it, so the same graph is
a different experiment at different clip lengths. That is a bad place to
begin an attribution.

*The instrumentation is not yet trustworthy, and this is the load-bearing
part.* The consumer's tracer does have per-call outcome attribution --
distinct values separating this fork's kernels from each fallback flavour
and from override delegation -- reachable at several exits. **No control
forces those outcomes.** They are correct-by-construction, with nothing
that would notice if one stopped landing where it should. A share
computed from an unforced attribution field is a plausible number, not
evidence, and it would be used to decide whether kernel work on this
model is worth doing at all. Closing that -- driving each outcome and
confirming the field follows -- is the prerequisite, and it is the same
defect class as everything in Findings A.

*One label is known wrong already:* calls are tagged with a fixed module
name although the patched method is shared by two block types, so a
shorter secondary workload arrives labelled as the primary one. It is
separable by sequence length, which is part of the record key, but not
by the field named for it. Anyone picking bench shapes from that trace
should split on length and ignore the label. This fork does not need to:
its own gated bench derives shapes from the node's geometry rules.

*Nothing currently joins the consumer's call counts to this fork's
dispatch counters.* Writing that join is a first, not a second
implementation -- but it belongs on the consumer side, since the
denominator does.

**D5. The bench comparand for the fused-MLP primitive is now one version
behind.** A newer scaled-matmul entry point exists in the installed
framework; our comparand uses the older one. Different tile selection
means this could move the primitive's verdict at the bench layer without
any kernel change. Low stakes -- that primitive ships as completeness,
not as a win -- but it is exactly the kind of comparand shift that
invalidates a stored conclusion. Already in `docs/roadmap.md`; this
audit only notes the version gate is now satisfied.

**D6. The newer-interpreter environment is built and unused.** Both
gates pass on it and every consumer node imports. What is untested is a
real render. Cutting over is cheap to try and cheap to revert, and the
only thing that would be learned by doing it is the thing no bench
covers.

**D7. What else has no re-check condition?** This audit found its
drifted claims by reading. The generalisation of R1 is to ask which
*other* statements in committed material would not fail if they became
false. The set is knowable: any claim about an external file, version,
or behaviour. That is a bounded sweep, and it has never been done.
