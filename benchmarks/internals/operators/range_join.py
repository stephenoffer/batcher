"""Benchmark the range (inequality) join against DuckDB, at controlled size and selectivity.

The range join is the one stateful operator with **no case in the committed suite**. Its
numbers have only ever been taken ad-hoc, into a docstring and a scorecard, and both have
since drifted: `competitive_architecture.md` ceiling 7 reports 1,493 ms at five million rows
a side while `bc-runtime/src/join/range/mod.rs` reports 522 ms for the same shape after a
rewrite neither document propagated. An operator nothing measures is an operator whose
regressions are invisible, which is precisely how the string sort lost for months.

This is that measurement, made repeatable.

**DuckDB must be given its own storage, and that is not a detail.** It picks `IE_JOIN` only
when its cardinality estimate for both inputs clears `merge_join_threshold` (default 1,000),
and a table registered from Arrow carries no such estimate — so it falls back to
`NESTED_LOOP_JOIN` and loses by two to three orders of magnitude. Reporting that would be
dishonest. Both tables are therefore ingested with an untimed ``CREATE TABLE`` and the plan is
asserted to contain `IE_JOIN` before any timing is trusted; the run fails loudly rather than
quietly measuring the wrong algorithm.

Three shapes, because the module implements three algorithms and they fail differently:

- ``one`` — a single inequality: the matches are a contiguous suffix of one sorted array.
- ``band`` — two inequalities bounding one shared right key (interval containment, temporal
  overlap, ``BETWEEN`` against a computed pair). The common real-world shape.
- ``ie`` — two independent inequalities: the general IEJoin (Khayyat et al., VLDB 2015), the
  algorithm DuckDB's ``PhysicalIEJoin`` implements.

Selectivity is held fixed as `n` grows rather than left to fall out of the data, because a
range join's cost is `O(n log n)` for its sorts plus `O(k)` for the pairs it emits: a shape
whose `k` grows quadratically measures the emit and tells you nothing about the algorithm.
Each shape's bounds are therefore sized from `n` so the expected match count per left row
stays constant, and the harness prints it so a reader can check.

Correctness first, as the suite insists: every shape's count is compared against DuckDB's
before any time is reported, and a mismatch aborts.

Run:
    source .venv/bin/activate
    python3 benchmarks/internals/operators/range_join.py                  # all shapes, 100K..2M
    python3 benchmarks/internals/operators/range_join.py --sizes 5000000  # the scorecard's largest
    python3 benchmarks/internals/operators/range_join.py --shape band
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

import batcher as bt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envinfo import machine_fingerprint, require_quiet_box, require_release_build

#: Matches per left row, held constant across `n` so the emit term does not swamp the sorts.
MATCHES_PER_LEFT_ROW = 4

#: The key universe. Wide enough that a band of the width computed below is narrow.
UNIVERSE = 1 << 30

SHAPES = ("one", "band", "ie")


def _tables(shape: str, n: int, seed: int = 17) -> tuple[pa.Table, pa.Table]:
    """Left and right tables of `n` rows whose match count per left row is ~constant in `n`."""
    rng = np.random.default_rng(seed)
    right_y = rng.integers(0, UNIVERSE, n, dtype=np.int64)
    if shape == "one":
        # `l.lo < r.y`: a suffix. Place each left bound so the suffix holds ~k rows.
        keep = max(1, MATCHES_PER_LEFT_ROW)
        lo = rng.integers(UNIVERSE - UNIVERSE * keep // n, UNIVERSE, n, dtype=np.int64)
        left = pa.table({"lo": pa.array(lo)})
    elif shape == "band":
        # `l.lo <= r.y <= l.hi`: a slice. Width sized so ~k of `n` uniform keys land inside.
        width = max(1, UNIVERSE * MATCHES_PER_LEFT_ROW // n)
        lo = rng.integers(0, UNIVERSE - width, n, dtype=np.int64)
        left = pa.table({"lo": pa.array(lo), "hi": pa.array(lo + width)})
    elif shape == "ie":
        # Two independent inequalities. Each alone keeps a fraction of the rows; together they
        # keep ~k, so each bound is placed at the square root of the target selectivity.
        frac = (MATCHES_PER_LEFT_ROW / n) ** 0.5
        edge = int(UNIVERSE * (1.0 - frac))
        lo = rng.integers(edge, UNIVERSE, n, dtype=np.int64)
        hi = rng.integers(0, UNIVERSE - edge, n, dtype=np.int64)
        left = pa.table({"lo": pa.array(lo), "hi": pa.array(hi)})
    else:  # pragma: no cover - argparse restricts this
        raise ValueError(shape)
    right_z = rng.integers(0, UNIVERSE, n, dtype=np.int64)
    right = pa.table({"y": pa.array(right_y), "z": pa.array(right_z)})
    return left, right


def _sql(shape: str) -> str:
    if shape == "one":
        return "SELECT count(*) AS n FROM l, r WHERE l.lo < r.y"
    if shape == "band":
        return "SELECT count(*) AS n FROM l, r WHERE l.lo <= r.y AND r.y <= l.hi"
    return "SELECT count(*) AS n FROM l, r WHERE l.lo < r.y AND l.hi > r.z"


def _duck(shape: str, left: pa.Table, right: pa.Table) -> tuple[duckdb.DuckDBPyConnection, str]:
    """A connection holding both tables in DuckDB's **native** store, plan asserted."""
    con = duckdb.connect()
    con.register("__l", left)
    con.register("__r", right)
    con.execute("CREATE TABLE l AS SELECT * FROM __l")
    con.execute("CREATE TABLE r AS SELECT * FROM __r")
    con.execute("ANALYZE")
    sql = _sql(shape)
    plan = con.execute(f"EXPLAIN {sql}").fetchall()[0][1]
    if shape != "one" and "IE_JOIN" not in plan:
        raise SystemExit(
            f"DuckDB planned {shape!r} without IE_JOIN; timing it would compare a different "
            f"algorithm. Plan was:\n{plan}"
        )
    return con, sql


def _best(fn, reps: int) -> float:
    """Milliseconds for the fastest of `reps` runs."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return min(times)


def run(shapes: tuple[str, ...], sizes: tuple[int, ...], reps: int) -> None:
    cols = ("shape", "n/side", "matches", "batcher", "duckdb", "b/duck")
    print(f"{cols[0]:6s} {cols[1]:>9s} {cols[2]:>13s} {cols[3]:>10s} {cols[4]:>10s} {cols[5]:>8s}")
    for shape in shapes:
        for n in sizes:
            left, right = _tables(shape, n)
            con, sql = _duck(shape, left, right)
            want = con.execute(sql).fetchone()[0]

            # Built once and collected repeatedly, so the timing is execution rather than
            # SQL parsing — the plan is lazy until `collect`.
            plan = bt.sql(sql, l=bt.from_arrow(left), r=bt.from_arrow(right))
            got = plan.collect().to_pydict()["n"][0]
            if got != want:
                raise SystemExit(f"{shape} n={n}: batcher {got} != duckdb {want}")

            b = _best(plan.collect, reps)
            d = _best(lambda c=con, q=sql: c.execute(q).fetchall(), reps)
            print(f"{shape:6s} {n:9,d} {want:13,d} {b:10.1f} {d:10.1f} {b / d:7.2f}x")


def main() -> None:
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
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shape", choices=SHAPES, action="append", help="repeatable; default all")
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100_000, 500_000, 1_000_000, 2_000_000],
        help="rows per side",
    )
    p.add_argument("--reps", type=int, default=3, help="timed repeats, best of")
    a = p.parse_args()
    run(tuple(a.shape or SHAPES), tuple(a.sizes), a.reps)


if __name__ == "__main__":
    main()
