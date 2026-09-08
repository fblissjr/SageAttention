#!/usr/bin/env python3
"""Assert the upstream facts this fork's design depends on.

Six claims in this repo's committed material went false during 2026 without
anything failing, and every one was found by a person reading rather than by a
check (`docs/drift_audit_and_directions.md`). Two of them were facts about the
consumer's source: a caller-side memory optimisation whose precondition had
disappeared, and a `path:line` pointer that no longer pointed at what it named.

This file is the answer to that. Each check names the documented claim it
guards, so a failure sends the reader to the sentence that needs editing rather
than to a debugger.

Three design rules, because this is easy to build wrong:

1. **Skip loudly, never silently.** The consumer stack is not a dependency of
   this fork. When it is absent this file says so and exits 0 with the skips
   printed. A skip that reads like a pass would reproduce the exact failure
   class it exists to end -- so the summary line always states how many checks
   actually ran.
2. **Assert the fact, not the line number.** Everything here parses source or
   inspects objects. A check that breaks on reformatting gets deleted by the
   next person, and then nothing is watching.
3. **Informational rows are printed, not asserted.** Some of these facts are
   not defects when they change -- they change what our entry points *buy*.
   Those print their finding and stay green. Only the load-bearing ones fail.

Deliberately parses rather than imports: importing ComfyUI pulls torch, CUDA
and a node registry, none of which this question needs.

    ${VIRTUAL_ENV}/bin/python tests/test_upstream_contracts.py
    ${VIRTUAL_ENV}/bin/python tests/test_upstream_contracts.py --comfy-root PATH
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


class Skip(Exception):
    """Raised when the consumer stack cannot be located. Never a failure."""


def comfy_root(cli: str | None) -> Path:
    """CLI arg > env var > internal/local_config.json > loud skip.

    The repo's documented resolution order for local-machine values. The final
    branch raises rather than guessing a path.
    """
    if cli:
        return Path(cli)
    env = os.environ.get("SAGE_COMFY_ROOT")
    if env:
        return Path(env)
    cfg = Path(__file__).resolve().parent.parent / "internal" / "local_config.json"
    if cfg.exists() and orjson is not None:
        root = orjson.loads(cfg.read_bytes()).get("comfyui_root")
        if root:
            return Path(root)
    raise Skip(
        "consumer stack not located. Pass --comfy-root, set SAGE_COMFY_ROOT, or "
        "add comfyui_root to internal/local_config.json (see the runbook)."
    )


def _parse(path: Path) -> ast.Module:
    if not path.exists():
        raise Skip(f"{path.name} not present in the consumer tree")
    return ast.parse(path.read_text())


# --- checks -----------------------------------------------------------------
# Each returns a note string on success. Raising AssertionError fails the file;
# raising Skip records a skip; returning a string starting with "INFO:" marks
# the check informational.

def check_h3_attention_passes_no_mask(root: Path) -> str:
    """GUARDS: CLAUDE.md, "mask=None hardcoded", and every claim that the
    v0.5.5 native-mask kernel is unreachable on H3.

    If this ever fails, H3 started passing masks and the mask kernel stops
    being dead weight -- which changes what is worth maintaining.
    """
    tree = _parse(root / "comfy" / "ldm" / "minimax" / "model.py")
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", getattr(n.func, "attr", None)) == "optimized_attention"
    ]
    assert calls, "no optimized_attention call found in the H3 model; the shape changed"
    masks = []
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert "mask" in kw, "an H3 attention call no longer passes mask explicitly"
        masks.append(isinstance(kw["mask"], ast.Constant) and kw["mask"].value is None)
    assert all(masks), (
        f"{masks.count(False)} of {len(masks)} H3 attention calls pass a non-None mask. "
        "CLAUDE.md says mask=None is hardcoded and that the v0.5.5 mask kernel is "
        "unreachable here; both claims now need revisiting."
    )
    return f"{len(calls)} attention call site(s), all mask=None"


def check_single_owner_container_protocol(root: Path) -> str:
    """GUARDS: docs/consumer_surface.md on `sageattn_consume` accepting three
    single-owner containers exposing peek()/take().

    If the protocol is renamed or loses a method, sageattn_consume's container
    path stops being reachable and nothing in our tests would notice.
    """
    tree = _parse(root / "comfy" / "ldm" / "modules" / "attention.py")
    cls = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.ClassDef) and n.name == "AttentionTensorContainer"),
        None,
    )
    assert cls is not None, (
        "AttentionTensorContainer is gone from comfy's attention module. "
        "sageattn_consume's container path depends on this protocol."
    )
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    missing = {"peek", "take"} - methods
    assert not missing, f"container protocol lost {sorted(missing)}"
    return "AttentionTensorContainer has peek() and take()"


def check_caller_clone_of_v(root: Path) -> str:
    """INFORMATIONAL. GUARDS: docs/consumer_surface.md on what
    `sageattn_consume` actually buys.

    Not a defect either way -- it changes what our entry point saves, not
    whether it is correct. This is the exact fact that silently went false:
    documented as cloning on 2026-08-11, found not cloning on 2026-09-08, and
    nothing failed in between.
    """
    tree = _parse(root / "comfy" / "ldm" / "minimax" / "model.py")
    clones = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "clone"
    ]
    if clones:
        return ("INFO: the H3 path calls .clone() somewhere -- if that is on v, "
                "sageattn_consume frees the fused buffer and the saving is live")
    return ("INFO: no .clone() on the H3 path, so q/k/v are views of one fused "
            "buffer and sageattn_consume saves nothing THERE. A consumer "
            "attention-patch node that clones v itself still gets the saving; "
            "the tracked one does, gated on our predicate.")


def check_masked_calls_gate_on_named_parameter(root: Path) -> str:
    """GUARDS: CLAUDE.md, "attn_mask must stay a named parameter of sageattn()".

    Our side is pinned by tests/test_dispatched_kernel_telemetry.py. This is
    the other half: that the consumer still gates on the signature at all. If
    they stop, our named-parameter constraint is no longer load-bearing and the
    doc should say so rather than implying a live dependency.
    """
    path = root / "comfy" / "ldm" / "modules" / "attention.py"
    src = _parse(path) and path.read_text()
    if "attn_mask" not in src:
        return ("INFO: no attn_mask reference in comfy's attention module; the "
                "named-parameter constraint may no longer be load-bearing")
    gates = "signature" in src or "getfullargspec" in src
    if not gates:
        return ("INFO: attn_mask appears but no signature introspection found. "
                "Re-read before relying on the named-parameter constraint.")
    return "consumer still introspects a signature around attn_mask"


CHECKS = [
    check_h3_attention_passes_no_mask,
    check_single_owner_container_protocol,
    check_caller_clone_of_v,
    check_masked_calls_gate_on_named_parameter,
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comfy-root", default=None)
    args = ap.parse_args()

    try:
        root = comfy_root(args.comfy_root)
    except Skip as exc:
        print(f"SKIP ALL: {exc}")
        print("\nran 0 of "
              f"{len(CHECKS)} checks. This is a SKIP, not a pass -- nothing "
              "about the consumer stack was verified.")
        return
    if not root.exists():
        print(f"SKIP ALL: consumer root {root} does not exist")
        print(f"\nran 0 of {len(CHECKS)} checks. This is a SKIP, not a pass.")
        return

    print(f"consumer stack: {root.name}/ (resolved)\n")
    ran = skipped = 0
    failures = []
    for c in CHECKS:
        name = c.__name__.replace("check_", "")
        try:
            note = c(root)
        except Skip as exc:
            skipped += 1
            print(f"SKIP  {name}: {exc}")
        except AssertionError as exc:
            ran += 1
            failures.append(name)
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report, never mask
            ran += 1
            failures.append(name)
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            ran += 1
            if note.startswith("INFO:"):
                print(f"INFO  {name}: {note[5:].strip()}")
            else:
                print(f"PASS  {name}: {note}")

    print()
    print(f"ran {ran} of {len(CHECKS)} checks ({skipped} skipped).")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        print("A failure here means a documented claim about upstream is now "
              "false. Fix the claim, then decide whether the code needs to "
              "follow.")
        sys.exit(1)
    print("No documented upstream claim contradicted.")


if __name__ == "__main__":
    main()
