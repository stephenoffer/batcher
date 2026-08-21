"""One plan, two schedulers: is the distributed result identical, and what does it cost?

The paper's central claim is that a stateful operator exists once and only its *scheduling*
varies, so single-node and distributed are the same program. That is asserted by a property
test over random partitionings and by one integration test. This runs it over a real query
suite instead: every TPC-H query executed twice on the same box, once through the
single-node executor and once through the distributed one, comparing results as row
multisets and reporting the tax distribution charges at a size where it buys nothing.

Running it at sf1 is the point. Distribution is the *wrong* choice there, so if the results
still agree the equivalence is not an artifact of a size that exercises the distributed path
gently.

Run:
    python benchmarks/scenarios/claims/scheduler_equivalence.py
    python benchmarks/scenarios/claims/scheduler_equivalence.py --scale 10
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Resolved from this file, not the working directory: the shared benchmark modules live
# in `benchmarks/`, two levels up. The cwd-relative form only imported when the script
# was launched from the repo root, and raised `ModuleNotFoundError: No module named
# 'context'` everywhere else -- including the spelling this file's own docstring gives.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import batcher as bt
from context import Context
from engines import resolve
from harness import results_match
from suites.standard.tpch import QUERIES


def _geomean(xs):
    xs = [x for x in xs if x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def _best(fn, rounds):
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args()

    ctx = Context.build("tpch", args.scale, resolve(["batcher"]))
    sess = bt.Session()
    for name in ctx.tables:
        sess.register(name, bt.from_arrow(ctx.table(name)))

    agree = mismatch = failed = 0
    ratios, notes = [], []
    print(f"\nOne plan, two schedulers - TPC-H sf{args.scale:g}, same box, best of {args.rounds}")
    hdr = f"{'case':<12}{'single (ms)':>13}{'dist (ms)':>12}{'tax':>9}  result"
    print(hdr)
    print("-" * (len(hdr) + 6))
    for case, sql in QUERIES.items():
        try:
            single = sess.sql(sql).collect()
        except Exception as exc:
            failed += 1
            notes.append(f"{case}: single-node raised {type(exc).__name__}")
            continue
        try:
            dist = sess.sql(sql).collect(distributed=True)
        except Exception as exc:
            failed += 1
            notes.append(f"{case}: distributed raised {type(exc).__name__}")
            continue
        ok, why = results_match(single, dist)
        if not ok:
            mismatch += 1
            print(f"{case:<12}{'':>13}{'':>12}{'':>9}  *** MISMATCH: {why[:44]}")
            continue
        agree += 1
        s_ms = _best(lambda q=sql: sess.sql(q).collect(), args.rounds)
        d_ms = _best(lambda q=sql: sess.sql(q).collect(distributed=True), args.rounds)
        ratios.append(d_ms / s_ms)
        print(f"{case:<12}{s_ms:>13.1f}{d_ms:>12.1f}{d_ms / s_ms:>8.1f}x  identical")
    print("-" * (len(hdr) + 6))
    print(f"identical: {agree}   mismatched: {mismatch}   not runnable: {failed}")
    if ratios:
        print(f"geomean distribution tax at this scale: {_geomean(ratios):.1f}x")
    for n in notes:
        print(f"  note: {n}")
    print("\nA mismatch here is a breach of the central invariant, not a slow query.")
    return 0 if mismatch == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
