#!/usr/bin/env python3
"""Pure-Python smoke tests for `check_regressions` in
`tests/test_sageattn_ltx_shapes.py`. Pinned baselines are a contract;
the regression-check logic that grades against them is the gate. A
silent breakage (e.g. the speedup-anchor lookup matching nothing
because shape names changed -- which actually shipped and was caught
by /simplify in v0.4.1) is exactly the bug class this file guards.

Standalone-script style per the rest of `tests/`: no pytest, no
fixtures. Run directly:
    ${VIRTUAL_ENV}/bin/python tests/test_regression_check.py

No CUDA, no sage, no torch needed at runtime -- pure data-flow
tests. The bench file imports torch at module load, so we must
import it last to fail fast with a clear message in venvs that
don't have torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import orjson


def _import_bench():
    """Import the bench module's check_regressions + Metrics. Lifts the
    import past torch's resolution so a missing-torch error fires
    immediately rather than buried inside a test."""
    import torch  # noqa: F401  -- import-only check; bench needs it

    sys.path.insert(0, str(Path(__file__).parent))
    import test_sageattn_ltx_shapes as bench
    return bench


def _baselines_path() -> Path:
    return Path(__file__).parent / "regression_baselines.json"


def _load_baselines() -> dict:
    return orjson.loads(_baselines_path().read_bytes())


def _measurements_from_baselines(bench, scale_ms: float = 1.0):
    """Build a fake measurements dict that mirrors what the bench would
    emit if every load-bearing baseline came back at exactly its
    pinned value (or scale_ms multiplied)."""
    cfg = _load_baselines()
    out = {}
    for entry in cfg.get("baselines", []):
        out[(entry["shape"], entry["mode"])] = bench.Metrics(
            mean_rtol=float(entry.get("mean_rtol") or 0.05),
            max_rtol=2.0,
            mean_atol=0.0001,
            max_atol=0.001,
            median_ms=float(entry["median_ms"]) * scale_ms,
        )
    return out


# --- tests --------------------------------------------------------------

def test_clean_baselines_match_themselves():
    """Mirror-image: feeding the baselines back as measurements must
    yield exit 0 and zero `regressions` (only `notes`).
    """
    bench = _import_bench()
    measurements = _measurements_from_baselines(bench, scale_ms=1.0)
    exit_code, lines = bench.check_regressions(measurements, _baselines_path())
    assert exit_code == 0, f"clean run failed: {lines}"
    fails = [l for l in lines if l.startswith(("PERF", "RTOL ", "MISSING", "SPEEDUP  "))
             and "below floor" in l]
    assert not fails, f"clean run produced regression lines: {fails}"


def test_speedup_line_appears():
    """The dead-branch class the v0.4.1 /simplify pass caught: the
    speedup-ratio block was unreachable because the shape lookup was
    hardcoded to a renamed string. With the auto-discovered anchor,
    every clean run must produce a SPEEDUP line in either
    `regressions` or `notes`. A silent absence = the anchor lookup
    is dead.
    """
    bench = _import_bench()
    measurements = _measurements_from_baselines(bench, scale_ms=1.0)
    _, lines = bench.check_regressions(measurements, _baselines_path())
    speedup_lines = [l for l in lines if l.startswith("SPEEDUP")]
    assert len(speedup_lines) == 1, (
        f"expected exactly one SPEEDUP line, got {len(speedup_lines)}: {speedup_lines}"
    )
    line = speedup_lines[0]
    assert "torch_flash/sage_fp8++" in line, (
        f"SPEEDUP line missing canonical ratio expression: {line}"
    )
    assert "x" in line and "floor" in line, (
        f"SPEEDUP line missing ratio + floor: {line}"
    )


def test_perf_drift_fails_load_bearing():
    """A 10% slowdown on a load-bearing row must trip exit_code=1
    (default perf_drift_pct=5.0).
    """
    bench = _import_bench()
    measurements = _measurements_from_baselines(bench, scale_ms=1.10)
    exit_code, lines = bench.check_regressions(measurements, _baselines_path())
    assert exit_code == 1, f"10% slowdown should fail, got exit {exit_code}: {lines}"
    perf_lines = [l for l in lines if l.startswith("PERF")]
    assert perf_lines, "no PERF lines on a 10% drift run"


def test_faster_is_informational_not_failure():
    """Going faster than baseline by > threshold prints a `FASTER`
    note but doesn't fail the gate -- regressions are one-directional.
    """
    bench = _import_bench()
    measurements = _measurements_from_baselines(bench, scale_ms=0.85)
    exit_code, lines = bench.check_regressions(measurements, _baselines_path())
    assert exit_code == 0, f"running faster shouldn't fail: {lines}"
    faster_lines = [l for l in lines if l.startswith("FASTER")]
    assert faster_lines, "expected FASTER notes on a 15% speedup"


def test_missing_load_bearing_fails():
    """Empty measurements dict -- every load-bearing baseline goes
    MISSING and the gate fails."""
    bench = _import_bench()
    exit_code, lines = bench.check_regressions({}, _baselines_path())
    assert exit_code == 1, "empty measurements must fail"
    missing = [l for l in lines if l.startswith("MISSING")]
    assert missing, "expected MISSING lines for an empty run"


def test_rtol_breach_fails():
    """A measurement above rtol_budget on a load-bearing row must
    trip exit_code=1.
    """
    bench = _import_bench()
    cfg = _load_baselines()
    rtol_budget = float(cfg.get("rtol_budget", 0.10))

    measurements = _measurements_from_baselines(bench, scale_ms=1.0)
    primary_key = ("ltx23_video_self_attn_init_22932", "fp8_cuda++")
    base = measurements[primary_key]
    measurements[primary_key] = bench.Metrics(
        mean_rtol=rtol_budget + 0.05,
        max_rtol=base.max_rtol,
        mean_atol=base.mean_atol,
        max_atol=base.max_atol,
        median_ms=base.median_ms,
    )
    exit_code, lines = bench.check_regressions(measurements, _baselines_path())
    assert exit_code == 1, f"rtol breach must fail: {lines}"
    rtol_lines = [l for l in lines if l.startswith("RTOL ")]
    assert rtol_lines, "expected RTOL lines on budget breach"


def test_missing_baselines_file_skips_gracefully():
    """A run with no baselines file at all: exit 0, single explanatory
    line, no crash. Lets the gate be opt-in -- bench works without
    pinned baselines configured."""
    bench = _import_bench()
    exit_code, lines = bench.check_regressions({}, Path("/nonexistent/baselines.json"))
    assert exit_code == 0, f"missing baselines should not fail: {lines}"
    assert len(lines) == 1 and "skipping" in lines[0], lines


# --- runner -------------------------------------------------------------

TESTS = [
    test_clean_baselines_match_themselves,
    test_speedup_line_appears,
    test_perf_drift_fails_load_bearing,
    test_faster_is_informational_not_failure,
    test_missing_load_bearing_fails,
    test_rtol_breach_fails,
    test_missing_baselines_file_skips_gracefully,
]


def main():
    failures = []
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures.append((test.__name__, str(exc)))
            print(f"FAIL  {test.__name__}: {exc}")
        except Exception as exc:
            failures.append((test.__name__, f"{type(exc).__name__}: {exc}"))
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {test.__name__}")
    print()
    if failures:
        print(f"FAIL {len(failures)}/{len(TESTS)}")
        sys.exit(1)
    print(f"PASS {len(TESTS)}/{len(TESTS)}")


if __name__ == "__main__":
    main()
