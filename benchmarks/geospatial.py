"""Throughput of the geospatial expressions, at three scales.

Answers one question: does per-row cost stay flat as the input grows. It should, because
every `ST_*` function is a scalar expression evaluated per row in Rust, but "should" is
not a measurement and the shape of this table is the only thing that distinguishes linear
scaling from something quietly quadratic.

Two things about the method, both of which a first attempt got wrong:

* **The reduction is over the expression's output, not `count()`.** A bare
  `select(...).count()` is answered from row-count metadata without evaluating anything,
  which produced a first draft reporting 1.3 billion rows/s and times that *fell* as the
  input grew. Aggregating the result forces the work.
* **Each case is run twice and the second run timed**, so the number excludes plan
  building and the first-touch page faults on a freshly built column.

There is no DuckDB comparison here. DuckDB's spatial extension is the correctness oracle
in `tests/differential/test_diff_geospatial.py`, but a timing comparison would be between
this engine's per-row interpreter and GEOS, which is a different claim than this file
makes and would need its own harness to be fair.

    python benchmarks/geospatial.py            # the default three scales
    python benchmarks/geospatial.py 5000000    # one explicit scale
"""

from __future__ import annotations

import sys
import time

import batcher as bt

#: A box around the sample points, so the predicate case is a real mix of hits and misses
#: rather than a uniform early reject.
BOX = "POLYGON((-122.45 37.7, -122.35 37.7, -122.35 37.8, -122.45 37.8, -122.45 37.7))"


def cases(ds):
    """The measured expressions, each reduced so nothing can be skipped."""
    point = bt.st_point(bt.col("lon"), bt.col("lat"))
    return [
        ("st_point -> st_area", lambda: ds.agg(s=bt.sum(bt.st_area(point))).to_pydict()),
        (
            "geohash_encode",
            lambda: ds.agg(
                s=bt.max(bt.geohash_encode(bt.col("lon"), bt.col("lat"), 8))
            ).to_pydict(),
        ),
        (
            "st_s2_cell",
            lambda: ds.agg(s=bt.max(bt.st_s2_cell(bt.col("lon"), bt.col("lat"), 15))).to_pydict(),
        ),
        (
            "st_intersects vs a box",
            lambda: ds.agg(s=bt.count_if(bt.st_intersects(point, bt.lit(BOX)))).to_pydict(),
        ),
        (
            "st_transform to UTM",
            lambda: ds.agg(s=bt.sum(bt.st_x(bt.st_transform(point, 4326, 32610)))).to_pydict(),
        ),
    ]


def run(n: int) -> None:
    """Time every case over `n` synthetic lon/lat rows."""
    lon = [-122.4 + (i % 1000) * 1e-4 for i in range(n)]
    lat = [37.7 + (i % 997) * 1e-4 for i in range(n)]
    ds = bt.from_pydict({"lon": lon, "lat": lat}).cache()
    ds.count()  # materialize the cache so the first case does not pay for it
    print(f"\n--- {n:,} rows ---")
    for label, call in cases(ds):
        call()  # warm: plan build and first touch
        start = time.perf_counter()
        call()
        elapsed = time.perf_counter() - start
        print(f"  {label:<24} {elapsed * 1000:8.1f} ms   {n / elapsed / 1e6:6.2f} M rows/s")


def main() -> None:
    scales = [int(a) for a in sys.argv[1:]] or [200_000, 800_000, 3_200_000]
    for n in scales:
        run(n)
    print(
        "\nRead the M rows/s column down each expression: flat means the per-row cost is "
        "independent of input size, which is what linear scaling looks like."
    )


if __name__ == "__main__":
    main()
