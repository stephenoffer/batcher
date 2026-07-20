#!/usr/bin/env python3
"""Measure benchmark timings as structured data, and fail on a regression.

The daily loop's `perf` pass may change hot-path code, so its gate has to answer two
questions mechanically: *is it still correct*, and *is it actually faster*. The benchmark
runner answers the first — it refuses to time a query whose result disagrees with DuckDB —
but it reports timings as a printed table, which a gate cannot compare against yesterday.

So this captures the same runs as a JSON snapshot and diffs two of them. It reuses the
harness's public pieces (`REGISTRY`, `Context`, `compare`) rather than re-implementing any
measurement, so a benchmark measured here is measured exactly as `benchmarks/run.py`
measures it — including the correctness check, which is the part that must not be
approximated. It deliberately does not edit `run.py`: a `--json` flag there would be the
tidier home, and is worth adding later, but that file was under concurrent edit.

Usage::

    python tools/agentic/bench.py --save before.json --benchmark tpch
    ...change something...
    python tools/agentic/bench.py --check before.json --benchmark tpch   # exit 1 on regression

A regression is a Batcher timing that got worse by more than `--tolerance` (default 10%,
which is above this harness's usual run-to-run noise at best-of-N). A query that stops being
correct is always a failure, regardless of timing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

#: The benchmark package imports its own modules flatly (`from harness import compare`),
#: so its directory has to be importable rather than the repo root.
sys.path.insert(0, str(REPO / "benchmarks"))

#: Default lineup: Batcher plus the two single-node engines the project benchmarks against.
DEFAULT_ENGINES = ("batcher", "duckdb", "polars")


def measure(
    benchmark: str,
    scale: float = 1.0,
    engine_names: tuple[str, ...] = DEFAULT_ENGINES,
    family: str | None = None,
    only: str | None = None,
    runs: int = 5,
) -> dict[str, Any]:
    """Run a benchmark and return its per-case, per-engine timings.

    Args:
        benchmark: Dataset name (`tpch`, `clickbench`, `operators`, ...).
        scale: Scale factor passed to the dataset builder.
        engine_names: Engines to run; Batcher must be among them to detect regressions.
        family: Restrict to one family, or None for all.
        only: Restrict to cases whose name contains this substring.
        runs: Best-of-N repetitions per case.

    Returns:
        A snapshot mapping each case name to its per-engine milliseconds and status.
    """
    import engines as engines_mod
    import suites  # noqa: F401  (import registers every benchmark case)
    from context import CORPUS_BENCHMARKS, Context
    from harness import compare
    from registry import REGISTRY

    engines = engines_mod.resolve([n.strip() for n in engine_names])
    names = [e.name for e in engines]
    cases = REGISTRY.select(dataset=benchmark, family=family, name=only)
    if not cases:
        raise SystemExit(f"no benchmark cases matched {benchmark!r} (family={family}, only={only})")

    builder = Context.build_corpus if benchmark in CORPUS_BENCHMARKS else Context.build
    ctx = builder(benchmark, scale, engines, None)

    snapshot: dict[str, Any] = {
        "benchmark": benchmark,
        "scale": scale,
        "engines": names,
        "runs": runs,
        "cases": {},
    }
    for case in cases:
        result = compare(case.name, case.build(ctx), names, runs=runs)
        snapshot["cases"][case.name] = {
            "status": result.status,
            "ms": {
                engine: result.engines[engine].ms
                for engine in names
                if engine in result.engines and result.engines[engine].ms is not None
            },
        }
    return snapshot


def regressions(
    before: dict[str, Any], after: dict[str, Any], tolerance: float = 0.10
) -> list[str]:
    """Return human-readable regressions between two snapshots.

    Both a correctness failure and a Batcher slowdown beyond ``tolerance`` count. A case
    that vanished between runs is reported too — silently dropping a benchmark is an easy
    way to make a regression disappear.
    """
    found: list[str] = []
    old_cases, new_cases = before.get("cases", {}), after.get("cases", {})

    for name in sorted(set(old_cases) - set(new_cases)):
        found.append(f"{name}: present in the baseline but not measured now")

    for name, new in sorted(new_cases.items()):
        if new["status"] != "OK":
            found.append(f"{name}: status {new['status']} (correctness failed or errored)")
            continue
        old = old_cases.get(name)
        if old is None:
            continue  # a new case has nothing to regress against
        before_ms, after_ms = old["ms"].get("batcher"), new["ms"].get("batcher")
        if before_ms is None or after_ms is None:
            continue
        if after_ms > before_ms * (1 + tolerance):
            pct = (after_ms / before_ms - 1) * 100
            found.append(f"{name}: batcher {before_ms:.1f}ms -> {after_ms:.1f}ms (+{pct:.1f}%)")
    return found


def improvements(
    before: dict[str, Any], after: dict[str, Any], tolerance: float = 0.10
) -> list[str]:
    """Return the cases that got meaningfully faster, for the report."""
    out: list[str] = []
    for name, new in sorted(after.get("cases", {}).items()):
        old = before.get("cases", {}).get(name)
        if not old or new["status"] != "OK":
            continue
        before_ms, after_ms = old["ms"].get("batcher"), new["ms"].get("batcher")
        if before_ms is None or after_ms is None:
            continue
        if after_ms < before_ms * (1 - tolerance):
            pct = (1 - after_ms / before_ms) * 100
            out.append(f"{name}: batcher {before_ms:.1f}ms -> {after_ms:.1f}ms (-{pct:.1f}%)")
    return out


def main() -> int:
    """Capture a benchmark snapshot, or check the current run against a saved one."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save", metavar="PATH", help="measure and write a snapshot")
    group.add_argument("--check", metavar="PATH", help="measure and compare against a snapshot")
    parser.add_argument("--benchmark", default="tpch")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--engines", default=",".join(DEFAULT_ENGINES))
    parser.add_argument("--family", default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="fractional slowdown tolerated before it counts as a regression",
    )
    args = parser.parse_args()

    snapshot = measure(
        args.benchmark,
        scale=args.scale,
        engine_names=tuple(args.engines.split(",")),
        family=args.family,
        only=args.only,
        runs=args.runs,
    )

    if args.save:
        Path(args.save).write_text(json.dumps(snapshot, indent=2, sort_keys=True))
        print(f"wrote {args.save} ({len(snapshot['cases'])} cases)")
        return 0

    baseline = json.loads(Path(args.check).read_text())
    faster = improvements(baseline, snapshot, args.tolerance)
    worse = regressions(baseline, snapshot, args.tolerance)

    for line in faster:
        print(f"  FASTER  {line}")
    if not worse:
        print(f"no regression across {len(snapshot['cases'])} case(s).")
        return 0
    print(f"REGRESSED — {len(worse)} case(s):")
    for line in worse:
        print(f"  {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
