"""Wall clock is not the only currency: CPU-seconds per query, Batcher vs DuckDB.

A wall-clock ratio on a 96-core box says how long you waited. It does not say what the
answer cost. An engine that spreads a query over four times the cores finishes sooner and
bills four times as much on a machine rented by the core-hour, and every ratio in
``BENCHMARK_RESULTS.md`` is blind to that.

Both engines run **in the same process**, so ``resource.getrusage(RUSAGE_SELF)`` charges
each one's threads to the same counter and the difference between two readings is that
engine's CPU time. That is what makes the comparison fair here and impossible against an
out-of-process engine (Spark, Ray) without per-cgroup accounting.

Read the CPU ratio, not the wall ratio, when the question is cost. Read ``cpu/wall`` as
effective parallelism: 1.0 is a serial query, 96.0 saturates this box.

**CPU time is the measurement least damaged by a busy machine.** Wall clock inflates under
contention; CPU-seconds are the work done either way. The gate still refuses to time a
query whose result does not match.

Run:
    python benchmarks/scenarios/claims/cost_bench.py                  # tpch sf1, 3 rounds
    python benchmarks/scenarios/claims/cost_bench.py --scale 10 --rounds 5
"""

from __future__ import annotations

import argparse
import math
import resource
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
from envinfo import machine_fingerprint, require_quiet_box, require_release_build
from harness import order_keys_of, order_violation, results_match
from suites.standard.tpch import QUERIES


def _cpu_seconds() -> float:
    """Process-wide CPU time (user + system), across every thread either engine spawns."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _time_once(fn: Any) -> tuple[float, float, Any]:
    """One execution: (wall ms, cpu ms, result)."""
    c0, w0 = _cpu_seconds(), time.perf_counter()
    out = fn()
    wall = (time.perf_counter() - w0) * 1000.0
    cpu = (_cpu_seconds() - c0) * 1000.0
    return wall, cpu, out


def _geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main() -> int:
    # Refuse to time a dev-profile engine: it is 8-60x slower, so a number taken from one
    # compares an unoptimized Batcher against release competitors. `BENCH_ALLOW_DEBUG_BUILD=1`
    # overrides deliberately.
    require_release_build()
    # Print the machine before any number: a timing is only reproducible beside the
    # box that produced it, and this file's own history has ratios quoted across four
    # different machines as if they were comparable.
    print(machine_fingerprint())
    # ...and refuse a contended one: a neighbour's load is not a fact about any
    # engine. `BENCH_ALLOW_BUSY_BOX=1` overrides.
    require_quiet_box()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    names = ["batcher", "duckdb"]
    engines = resolve(names)
    for e in engines:
        if not e.available():
            print(f"engine {e.name} unavailable")
            return 1

    ctx = Context.build("tpch", args.scale, engines)
    runners = ctx.sql_runners()

    rows: list[tuple[str, float, float, float, float]] = []
    skipped: list[str] = []
    for case, sql in QUERIES.items():
        # Correctness gate before any timing, exactly as the main harness does.
        try:
            out = {n: runners[n](sql) for n in names}
        except Exception as exc:  # an engine that cannot run this case
            skipped.append(f"{case} (raised: {type(exc).__name__})")
            continue
        ok, why = results_match(out["duckdb"], out["batcher"])
        # The multiset comparison sorts both sides, so an engine that skipped its `ORDER BY`
        # would pass it and then be charged only for the work it did. A CPU-cost comparison
        # is exactly where that matters: the skipped sort is the cost.
        if ok:
            keys = order_keys_of(sql)
            for engine, table in out.items():
                violation = order_violation(table, keys)
                if violation is not None:
                    ok, why = False, f"{engine} {violation}"
                    break
        if not ok:
            skipped.append(f"{case} (MISMATCH: {why})")
            continue

        best: dict[str, tuple[float, float]] = {}
        for n in names:
            runners[n](sql)  # warm-up, untimed

            def call(n=n, q=sql):
                return runners[n](q)

            samples = [_time_once(call)[:2] for _ in range(args.rounds)]
            # Minimum of each independently: wall is the best-case wait, CPU the
            # least-polluted estimate of the work.
            best[n] = (min(s[0] for s in samples), min(s[1] for s in samples))
        bw, bc = best["batcher"]
        dw, dc = best["duckdb"]
        rows.append((case, bw / dw, bc / dc, bc / bw, dc / dw))

    hdr = f"{'case':<12}{'wall b/d':>10}{'cpu b/d':>10}{'batcher':>10}{'duckdb':>10}"
    print(f"\nTPC-H sf{args.scale:g}, best of {args.rounds}, both engines in one process")
    print(f"{'':<12}{'':>10}{'':>10}{'cpu/wall':>10}{'cpu/wall':>10}")
    print(hdr)
    print("-" * len(hdr))
    for case, wr, cr, bpar, dpar in rows:
        print(f"{case:<12}{wr:>10.2f}{cr:>10.2f}{bpar:>10.1f}{dpar:>10.1f}")
    print("-" * len(hdr))
    print(
        f"{'geomean':<12}"
        f"{_geomean([r[1] for r in rows]):>10.2f}"
        f"{_geomean([r[2] for r in rows]):>10.2f}"
        f"{_geomean([r[3] for r in rows]):>10.1f}"
        f"{_geomean([r[4] for r in rows]):>10.1f}"
    )
    print(f"\n{len(rows)} queries gated and timed; skipped: {skipped or 'none'}")
    print("Below 1.00 is Batcher cheaper. cpu/wall is effective cores used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
