"""How to time a query without fooling yourself.

Four rules: warm the cache, run more than once, verify the result, and report the shape of
the distribution rather than the best number. The last one matters most — a minimum is not a
measurement, it is the luckiest sample.

    python examples/perf/measuring_honestly.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch_path
from batcher import col


def main() -> None:
    path = tpch_path("lineitem")

    def query():
        return (
            bt.read.parquet(path)
            .filter(col("l_quantity") > 25)
            .group_by("l_shipmode")
            .agg(revenue=col("l_extendedprice").sum())
            .sort("l_shipmode")
            .to_pydict()
        )

    # Rule 3 first: verify before timing anything. A timing on an unverified path is worth
    # nothing at all.
    result = query()
    assert len(result["l_shipmode"]) > 0
    assert all(value > 0 for value in result["revenue"])
    expected_lines = bt.read.parquet(path).filter(col("l_quantity") > 25).count()
    assert expected_lines > 0

    # Rule 1: warm up.
    query()

    # Rule 2: several runs.
    timings: list[float] = []
    for _ in range(7):
        started = time.perf_counter()
        run = query()
        timings.append((time.perf_counter() - started) * 1000)
        # Rule 3 again: every run is checked, not just the first.
        assert run == result

    # Rule 4: report the distribution.
    timings.sort()
    print(f"runs:    {[round(value, 1) for value in timings]}")
    print(f"min      {timings[0]:7.1f} ms")
    print(f"median   {statistics.median(timings):7.1f} ms")
    print(f"max      {timings[-1]:7.1f} ms")
    spread = timings[-1] - timings[0]
    print(f"spread   {spread:7.1f} ms ({spread / statistics.median(timings):.1%} of median)")

    assert len(timings) == 7
    assert timings[0] <= statistics.median(timings) <= timings[-1]
    assert all(value > 0 for value in timings)

    # The minimum is always optimistic relative to the median, which is why quoting it
    # alone overstates the result.
    assert timings[0] <= statistics.median(timings)


if __name__ == "__main__":
    main()
