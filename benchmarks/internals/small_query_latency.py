"""Fixed per-query latency of the control plane, isolated from the data.

A query over a wide table costs the same whether it reads a thousand rows or a million:
the work that dominates it is proportional to the *source's column count*, not to the
data. This benchmark measures that fixed cost directly, by running one query at several
column counts and several row counts, so a change to the orchestration can be held to a
number instead of a profile reading.

Why it exists: on ClickBench (105 columns) every query pays this before the engine is
reached, and the queries that benchmark reports in single-digit milliseconds are decided
by it. `api/orchestration/fast_path.py` documents the same effect from the other side --
it skips the orchestration and measures what is left.

Run:
    python benchmarks/internals/small_query_latency.py
"""

from __future__ import annotations

import dataclasses
import statistics
import sys
import time

import pyarrow as pa

import batcher as bt
from batcher.config import active_config, set_config

#: Column counts to sweep. 105 is ClickBench's `hits`; 16 is TPC-H `lineitem`.
COLUMN_COUNTS = (1, 8, 16, 40, 105)
#: Rows in the probe table. Small on purpose -- the point is the cost that is *not* the data.
PROBE_ROWS = 20_000
_WARMUP = 10
_REPEATS = 60


def _probe_table(ncols: int, nrows: int) -> pa.Table:
    """A table of `ncols` int64 columns, deterministic and cheap to build."""
    col = pa.array(range(nrows), type=pa.int64())
    return pa.table({f"c{i}": col for i in range(ncols)})


def _best_ms(fn, repeats: int = _REPEATS) -> float:
    """Best-of-N wall time in milliseconds (best, not mean: this is a latency floor)."""
    for _ in range(_WARMUP):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return min(samples)


def _median_ms(fn, repeats: int = _REPEATS) -> float:
    """Median wall time in milliseconds, for the noise-tolerant companion figure."""
    for _ in range(_WARMUP):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def measure(ncols: int, nrows: int = PROBE_ROWS) -> tuple[float, float]:
    """Best and median `collect()` latency for a one-column filter over `ncols` columns."""
    session = bt.Session()
    session.register("t", _probe_table(ncols, nrows))
    ds = session.sql("SELECT c0 FROM t WHERE c0 = 7")
    return _best_ms(lambda: ds.collect()), _median_ms(lambda: ds.collect())


def main() -> int:
    """Sweep the column counts and print the per-column slope."""
    cfg = active_config()
    # Match the benchmark suite: the event log is an observability write, not engine work.
    set_config(cfg.replace(observability=dataclasses.replace(cfg.observability, event_log=False)))

    print(f"fixed per-query latency, {PROBE_ROWS:,}-row probe, best-of-{_REPEATS}\n")
    print(f"{'columns':>8}  {'best_ms':>8}  {'median_ms':>10}")
    print("-" * 30)
    points = []
    for ncols in COLUMN_COUNTS:
        best, median = measure(ncols)
        points.append((ncols, best))
        print(f"{ncols:>8}  {best:>8.3f}  {median:>10.3f}")

    (lo_cols, lo_ms), (hi_cols, hi_ms) = points[0], points[-1]
    per_column = (hi_ms - lo_ms) / (hi_cols - lo_cols)
    print(f"\nper-column slope: {per_column * 1000:.1f} us/column")
    print(f"intercept (1 column): {lo_ms:.3f} ms")

    # Row-independence: the same query over 50x the rows must cost the same fixed part.
    wide_small, _ = measure(105, 20_000)
    wide_big, _ = measure(105, 1_000_000)
    print(f"\n105 columns @ 20k rows: {wide_small:.3f} ms")
    print(f"105 columns @ 1M rows:  {wide_big:.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
