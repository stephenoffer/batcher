"""Telling a scan-bound query from a compute-bound one.

The diagnosis is one comparison: how long does the same scan take with almost no work on top
of it. If adding the computation barely changes the time, the query is scan-bound and no
amount of expression tuning will help.

    python examples/perf/scan_vs_compute_bound.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch_path
from batcher import col


def main() -> None:
    path = tpch_path("lineitem")

    def timed(label: str, build) -> float:
        # Warm the page cache first so the comparison is about work, not about IO luck.
        build()
        started = time.perf_counter()
        build()
        elapsed = (time.perf_counter() - started) * 1000
        print(f"{label:<34} {elapsed:7.1f} ms")
        return elapsed

    bare = timed("count only", lambda: bt.read.parquet(path).count())
    one_column = timed(
        "one column summed",
        lambda: bt.read.parquet(path).agg(t=col("l_quantity").sum()).to_pydict(),
    )
    heavy = timed(
        "arithmetic over three columns",
        lambda: (
            bt.read.parquet(path)
            .agg(t=(col("l_extendedprice") * (1 - col("l_discount")) * (1 + col("l_tax"))).sum())
            .to_pydict()
        ),
    )
    string_work = timed(
        "substring match over comments",
        lambda: bt.read.parquet(path).filter(col("l_comment").str.contains("final")).count(),
    )

    for label, value in (
        ("count", bare),
        ("one column", one_column),
        ("arithmetic", heavy),
        ("string", string_work),
    ):
        assert value > 0, label

    # The string scan is the compute-bound one: it must look at every byte of a text
    # column that no other query here reads.
    print(
        f"string work costs {string_work / max(one_column, 1e-9):.1f}x the single-column aggregate"
    )
    assert string_work > 0

    # And the answers are what they should be, because a timing on an unverified path is
    # worth nothing.
    assert bt.read.parquet(path).count() == 200_000
    assert bt.read.parquet(path).filter(col("l_comment").str.contains("final")).count() > 0


if __name__ == "__main__":
    main()
