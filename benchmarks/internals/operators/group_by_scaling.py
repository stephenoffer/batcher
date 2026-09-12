"""How does a `GROUP BY` cost scale in the *group count*, and in the *aggregate count*?

A single-size group-by timing hides the two things that decide whether this operator is
competitive at scale, and they are fixed by different work:

* **Scaling in groups.** Measured on a 16-core box over 24 M rows: 1.66x DuckDB at 100 groups,
  falling to **1.09x at 10 M** -- the ratio *improves* with cardinality, and Batcher is 2.8x
  ahead of Polars at the top of the range. There is no scaling gap here.
* **Scaling in aggregates.** At 11.5 M groups DuckDB is flat at 218-226 ms whether it is asked
  for `count(*)` or five aggregates. Batcher is **faster than DuckDB on the bare grouping**
  (155 ms against 226) and adds ~54 ms per additional aggregate, reaching 1.66x at five. That
  per-aggregate cost, which DuckDB does not pay at all, is the one real gap this benchmark
  shows.

Those two ladders are what this benchmark reports, because the split between them is what says
where to work. See `BENCHMARK_RESULTS.md`, 2026-09-08.

**The first version of this file reported 5.37x and 8.19x for the same two ladders**, and it
was measuring the streaming executor rather than the aggregate: every arm folds the group-by to
a scalar so the engines can be compared on one number, and a global aggregate above a grouped
one used to make Kyber's routing peel answer "not a grouped aggregate". Both peels were
extended and the ladders moved by 5x without a line of the aggregate changing. The arms still
fold to a scalar -- the point is that they now measure what they claim to.

**The fixture is chunked on purpose and that is load-bearing.** DuckDB parallelises its Arrow
scan per record batch, so a `pa.table({...})` built over whole numpy arrays -- one 24 M-row
chunk per column -- runs it effectively single-threaded and costs it ~5x. Batcher's own time
does not move with chunking (measured: 1,867-1,946 ms across four chunkings of the same data),
so a one-chunk fixture silently times a handicapped comparator and inverts the result. An
earlier revision of this measurement did exactly that and reported Batcher *ahead* at three of
four cardinalities.

Run:
    python benchmarks/internals/operators/group_by_scaling.py            # both ladders
    python benchmarks/internals/operators/group_by_scaling.py 8000000    # a different row count
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envinfo import machine_fingerprint, require_release_build

#: Chunks the fixture is cut into. Any number that gives the comparator's scan something to
#: parallelise over will do; this is the shape a four-file sf100 Parquet read produces.
_CHUNKS = 190

#: Group counts to sweep. The top of the range is where the scaling gap is visible; the bottom
#: is the control that shows it is a scaling gap and not a constant one.
_CARDINALITIES = (100, 100_000, 2_000_000, 11_500_000)


def _fixture(rows: int, ndv: int) -> pa.Table:
    """A chunked `(k, v, q)` table with `ndv` distinct keys drawn uniformly.

    Args:
        rows: Total rows.
        ndv: Distinct values the key is drawn from.

    Returns:
        The fixture, cut into `_CHUNKS` chunks per column.
    """
    rng = np.random.default_rng(3)
    keys = rng.integers(0, ndv, rows).astype("int64")
    vals = rng.random(rows) * 1000.0
    qty = rng.integers(1, 50, rows).astype("int64")
    return pa.table(
        {
            "k": pa.chunked_array(np.array_split(keys, _CHUNKS)),
            "v": pa.chunked_array(np.array_split(vals, _CHUNKS)),
            "q": pa.chunked_array(np.array_split(qty, _CHUNKS)),
        }
    )


def _best(fn, repeats: int = 3) -> tuple[float, object]:
    """Fastest of `repeats` runs, in milliseconds, with that run's result.

    Args:
        fn: The zero-argument call to time.
        repeats: How many times to run it.

    Returns:
        `(milliseconds, result)` for the fastest run.
    """
    times, out = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - start)
    return min(times) * 1000.0, out


_FIVE_SQL = (
    "SELECT sum(s), count(*) FROM"
    " (SELECT k, sum(v) s, sum(q), max(v), min(v), count(*) FROM t GROUP BY k)"
)


def _five_aggregates(ds):
    """The five-aggregate group-by, folded to one row so the comparison is a scalar."""
    from batcher import col, count

    return (
        ds.group_by("k")
        .agg(s=col("v").sum(), q2=col("q").sum(), mx=col("v").max(), mn=col("v").min(), n=count())
        .agg(t=col("s").sum(), g=count())
        .collect()
        .to_pydict()
    )


def _five_polars(df):
    """Polars' spelling of the same query. Every aggregate needs its own alias, or two
    expressions over `v` collide on the name and polars refuses the plan."""
    import polars as pl

    grouped = df.group_by("k").agg(
        pl.col("v").sum().alias("s"),
        pl.col("q").sum().alias("q2"),
        pl.col("v").max().alias("mx"),
        pl.col("v").min().alias("mn"),
        pl.len().alias("n"),
    )
    return grouped.select(pl.col("s").sum().alias("t"), pl.len().alias("g")).to_dicts()


def sweep_cardinality(rows: int) -> None:
    """Time the same five-aggregate group-by at every cardinality in `_CARDINALITIES`.

    Args:
        rows: Rows in each fixture.
    """
    import duckdb
    import polars as pl

    import batcher as bt

    print(f"\n{'groups':>12} | {'batcher':>10} | {'duckdb':>10} | {'polars':>10} | ratio b/d")
    for ndv in _CARDINALITIES:
        tbl = _fixture(rows, ndv)
        ds = bt.from_arrow(tbl)
        con = duckdb.connect()
        con.register("t", tbl)
        df = pl.from_arrow(tbl)

        bat_ms, bat = _best(lambda d=ds: _five_aggregates(d))
        duck_ms, duck = _best(lambda c=con: c.execute(_FIVE_SQL).fetchall())
        pol_ms, _ = _best(lambda f=df: _five_polars(f))

        groups = int(bat["g"][0])
        agree = "AGREE" if groups == int(duck[0][1]) else f"DIFFER {groups} vs {duck[0][1]}"
        ratio = bat_ms / duck_ms
        print(
            f"{groups:>12,} | {bat_ms:9.1f}ms | {duck_ms:9.1f}ms | {pol_ms:9.1f}ms |"
            f" {ratio:5.2f}x  {agree}"
        )


def _ladder(ds):
    """`label -> (batcher thunk, equivalent SQL)` over the same grouping, 0 to 5 aggregates."""
    from batcher import col, count

    return {
        "count(*) only": (
            lambda: ds.group_by("k").agg(n=count()),
            "SELECT k, count(*) FROM t GROUP BY k",
        ),
        "1 sum": (
            lambda: ds.group_by("k").agg(s=col("v").sum()),
            "SELECT k, sum(v) FROM t GROUP BY k",
        ),
        "3 aggregates": (
            lambda: ds.group_by("k").agg(s=col("v").sum(), q2=col("q").sum(), n=count()),
            "SELECT k, sum(v), sum(q), count(*) FROM t GROUP BY k",
        ),
        "5 aggregates": (
            lambda: ds.group_by("k").agg(
                s=col("v").sum(),
                q2=col("q").sum(),
                mx=col("v").max(),
                mn=col("v").min(),
                n=count(),
            ),
            "SELECT k, sum(v), sum(q), max(v), min(v), count(*) FROM t GROUP BY k",
        ),
    }


def sweep_aggregates(rows: int, ndv: int = 11_500_000) -> None:
    """Time the same grouping with 0, 1, 3 and 5 value aggregates over it.

    Args:
        rows: Rows in the fixture.
        ndv: Distinct keys, i.e. how many groups the aggregates are spread over.
    """
    import duckdb

    import batcher as bt
    from batcher import count

    tbl = _fixture(rows, ndv)
    ds = bt.from_arrow(tbl)
    con = duckdb.connect()
    con.register("t", tbl)

    print(f"\n{'aggregates':>14} | {'batcher':>10} | {'duckdb':>10} | ratio  (at {ndv:,} groups)")
    for label, (build, sql) in _ladder(ds).items():
        bat_ms, bat = _best(lambda b=build: b().agg(g=count()).collect().to_pydict())
        duck_ms, duck = _best(
            lambda c=con, q=sql: c.execute(f"SELECT count(*) FROM ({q})").fetchall()
        )
        agree = "AGREE" if int(bat["g"][0]) == int(duck[0][0]) else "DIFFER"
        ratio = bat_ms / duck_ms
        print(f"{label:>14} | {bat_ms:9.1f}ms | {duck_ms:9.1f}ms | {ratio:5.2f}x  {agree}")


def main() -> None:
    """Run both ladders and print the machine they were taken on."""
    require_release_build()
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 24_000_000
    print(machine_fingerprint())
    print(f"rows: {rows:,}, fixture cut into {_CHUNKS} chunks")
    sweep_cardinality(rows)
    sweep_aggregates(rows)


if __name__ == "__main__":
    main()
