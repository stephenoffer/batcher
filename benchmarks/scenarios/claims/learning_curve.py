"""Does the engine actually get faster the second time? The curve, with a control.

The cross-query learning loop is the thing this system claims that no comparator has, and
it has been supported by a single before/after number. This measures the whole curve: run
each query K times back to back in one **fresh** process (the learned store is
``in_process``, so a new process is a cold hub) and report execution *n* against the
steady state.

The control is the point. "Query 2 is faster than query 1" is also what a page cache, a
JIT and a malloc arena produce, so on its own it proves nothing. DuckDB runs the identical
loop on the identical data in the same process and has no cross-query learning: whatever
its curve does is the environmental floor, and only the part of Batcher's curve that
exceeds it can be attributed to the loop.

Correctness is gated on the first execution of every query before any timing is kept.

Run:
    python benchmarks/scenarios/claims/learning_curve.py                 # tpch sf1, 6 executions
    python benchmarks/scenarios/claims/learning_curve.py --runs 10 --scale 10
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

# Resolved from this file, not the working directory: the shared benchmark modules live
# in `benchmarks/`, two levels up. The cwd-relative form only imported when the script
# was launched from the repo root, and raised `ModuleNotFoundError: No module named
# 'context'` everywhere else -- including the spelling this file's own docstring gives.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from context import Context
from engines import resolve
from harness import results_match
from suites.standard.tpch import QUERIES


def _geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def _curve(run: Any, sql: str, runs: int) -> list[float]:
    """Wall time (ms) of executions 1..runs, back to back, no warm-up."""
    out = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run(sql)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--runs", type=int, default=6)
    args = ap.parse_args()

    names = ["batcher", "duckdb"]
    engines = resolve(names)
    ctx = Context.build("tpch", args.scale, engines)
    runners = ctx.sql_runners()

    curves: dict[str, list[list[float]]] = {n: [] for n in names}
    skipped = []
    for case, sql in QUERIES.items():
        try:
            first = {n: runners[n](sql) for n in names}
        except Exception as exc:
            skipped.append(f"{case} ({type(exc).__name__})")
            continue
        ok, why = results_match(first["duckdb"], first["batcher"])
        if not ok:
            skipped.append(f"{case} (MISMATCH: {why})")
            continue
        for n in names:
            curves[n].append(_curve(lambda q, n=n: runners[n](q), sql, args.runs))

    print(
        f"\nLearning curve — TPC-H sf{args.scale:g}, {args.runs} back-to-back executions, "
        f"fresh process (cold learned store)"
    )
    print("Each cell is execution n / that engine's own steady state (last execution).")
    hdr = f"{'engine':<10}" + "".join(f"{'run ' + str(i + 1):>9}" for i in range(args.runs))
    print(hdr)
    print("-" * len(hdr))
    norm: dict[str, list[float]] = {}
    for n in names:
        per_run = []
        for i in range(args.runs):
            per_run.append(_geomean([c[i] / c[-1] for c in curves[n] if c[-1] > 0]))
        norm[n] = per_run
        print(f"{n:<10}" + "".join(f"{v:>9.2f}" for v in per_run))
    print("-" * len(hdr))
    excess = [norm["batcher"][i] / norm["duckdb"][i] for i in range(args.runs)]
    print(f"{'excess':<10}" + "".join(f"{v:>9.2f}" for v in excess))
    print("\n'excess' is Batcher's curve divided by DuckDB's — the part not explained by")
    print("page cache, JIT or allocator warming, since DuckDB has no cross-query learning.")
    print(f"{len(curves['batcher'])} queries; skipped: {skipped or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
