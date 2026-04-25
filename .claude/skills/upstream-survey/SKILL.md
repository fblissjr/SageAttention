---
name: upstream-survey
description: Run the quarterly upstream survey for sage-fork. Clones thu-ml/SageAttention and woct0rdho/SageAttention into /tmp, lists recent commits, diffs the surfaces this fork tracks (sageattention/, csrc/), and filters out noise we don't care about (Windows-only, sm75/sm87, sage3 Blackwell reorganization, README/example updates). Use when the user says "run upstream survey", "check upstream", "survey thu-ml and woct0rdho", or when the quarterly cron-like check is due. Read-only — does not modify the working tree.
disable-model-invocation: true
---

# Upstream survey

last updated: 2026-04-25

This skill mechanizes the quarterly survey documented in
`CHANGELOG.md` / "Recurring process items" / "Periodic upstream
survey." Read that section first if you want the full context; the
short version is below.

## Why this exists

We squashed history at 2026-04-23 and stopped pulling from either
upstream. Two risks remain:

1. `thu-ml/SageAttention` lands a kernel bugfix via PR and we miss it.
2. `woct0rdho/SageAttention` ships a future regression (likely in
   Windows-focused work) that silently drifts us if we ever
   cherry-pick from them.

This skill catches both classes by diffing recent upstream state
against our tree, with the noise filtered out.

## What to look for (act on these)

- New `.cu` / `.cuh` fixes in `csrc/qattn/`, `csrc/fused/`,
  `csrc/utils.cuh` (and adjacent attention-kernel headers).
- Changes to `sageattention/core.py` dispatch logic (new arch branch,
  new pv_accum variant, changed routing).
- `setup.py` arch-gate changes (the gate at line 152 is the load-bearing
  reason this fork exists).

## What to ignore (note in passing, don't act)

- Windows-specific changes (path separators, MSVC pragmas, .bat scripts).
- sm75/sm87/sm120 specific changes (we only care about sm89).
- `sageattention3_blackwell/` reorganizations (not on our path).
- README / example / docs-only updates.
- Wheel build matrix changes.

## Workflow

### Step 1 — Confirm we're at the sage-fork repo root

```bash
test -f ./CLAUDE.md && grep -q '^# sage-fork' ./CLAUDE.md && echo "ok" || \
  { echo "error: run from sage-fork repo root"; exit 1; }
```

### Step 2 — Clone (or refresh) the two upstreams into /tmp

Shallow clones; we only want recent history.

```bash
cd /tmp && rm -rf sage-survey-thuml sage-survey-woct
git clone --depth 50 --quiet https://github.com/thu-ml/SageAttention sage-survey-thuml
git clone --depth 50 --quiet https://github.com/woct0rdho/SageAttention sage-survey-woct
```

### Step 3 — Recent commits

Default window is 3 months; widen if the prior survey was longer ago
(check `internal/log/` for the most recent survey entry).

```bash
SINCE="${SINCE:-3 months ago}"
echo "=== thu-ml since $SINCE ==="
git -C /tmp/sage-survey-thuml log --oneline --since="$SINCE"
echo "=== woct0rdho since $SINCE ==="
git -C /tmp/sage-survey-woct log --oneline --since="$SINCE"
```

### Step 4 — Surface diffs (run from sage-fork repo root)

```bash
echo "=== thu-ml vs us: sageattention/ ==="
diff -qr /tmp/sage-survey-thuml/sageattention ./sageattention 2>&1 | \
  grep -vE '__pycache__|\.pyc$'
echo "=== thu-ml vs us: csrc/ ==="
diff -qr /tmp/sage-survey-thuml/csrc ./csrc

echo "=== woct0rdho vs us: sageattention/ ==="
diff -qr /tmp/sage-survey-woct/sageattention ./sageattention 2>&1 | \
  grep -vE '__pycache__|\.pyc$'
echo "=== woct0rdho vs us: csrc/ ==="
diff -qr /tmp/sage-survey-woct/csrc ./csrc
```

### Step 5 — Synthesize a verdict

For each diff that surfaced, classify:

- **Act now**: kernel `.cu`/`.cuh` bugfix on a path we use (sm80/sm89
  qattn, fused), dispatch logic change in `core.py`, setup.py arch-gate
  change.
- **Note for next survey**: anything else that's not in the ignore list.
- **Ignore**: matches the ignore list above.

### Step 6 — Record the survey outcome

Append to `internal/log/log_<today>.md` under a new section:

```markdown
## Upstream survey — <YYYY-MM-DD>

Window: <N> months since previous survey on <prior date>.

thu-ml:
- <N commits in window>
- act-now items: <list or "none">
- noted: <list or "none">

woct0rdho:
- <N commits in window>
- act-now items: <list or "none">
- noted: <list or "none">

Next survey: <YYYY-MM-DD> (3 months out)
```

If any "act now" items appear, file them under `CHANGELOG.md` /
"Backlog" with an explicit trigger-to-act, then close out the survey.

## Baseline (do not re-derive)

As of 2026-04-24 (squash time):

- woct0rdho's `core.py` had meaningful improvements over thu-ml
  (CUDA-version-gated fp8++ dispatch, `torch.version.cuda` detection,
  `_cuda_archs` cache). We kept these.
- thu-ml had nothing we wanted at that time.

Future surveys diff against THIS state (the squash baseline + our
layered changes), not the full upstream history. The shallow `--depth 50`
clone is sufficient for the quarterly window.

## Cleanup

The /tmp clones are scratch; leave them or delete:

```bash
rm -rf /tmp/sage-survey-thuml /tmp/sage-survey-woct
```
