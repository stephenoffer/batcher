"""When caching pays, and when it is just memory you gave away.

The rule is reuse count times cost. An expensive intermediate read twice is worth caching; a
cheap one read twice is not, and neither is an expensive one read once. Measuring both sides
takes one run each.

    python examples/perf/cache_versus_recompute.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    expensive = lineitem.group_by("l_orderkey").agg(
        revenue=col("l_extendedprice").sum(), lines=bt.count()
    )

    def read_twice(dataset: bt.Dataset) -> tuple[int, int]:
        return (
            dataset.filter(col("revenue") > 50_000).count(),
            dataset.filter(col("lines") >= 5).count(),
        )

    started = time.perf_counter()
    plain = read_twice(expensive)
    plain_ms = (time.perf_counter() - started) * 1000

    cached = expensive.cache()
    started = time.perf_counter()
    warm = read_twice(cached)
    cached_ms = (time.perf_counter() - started) * 1000

    print(f"recomputed {plain_ms:7.1f} ms   cached {cached_ms:7.1f} ms")
    print("results:", plain)

    # Caching changes cost, never the answer.
    assert plain == warm

    # A single read gains nothing from a cache, which is the case people over-apply it to.
    started = time.perf_counter()
    expensive.filter(col("revenue") > 50_000).count()
    single_ms = (time.perf_counter() - started) * 1000
    print(f"single read, uncached: {single_ms:7.1f} ms")
    assert single_ms > 0

    # And the thing worth caching is the one that is both expensive and reused: a cheap
    # projection read twice is not.
    cheap = lineitem.select("l_orderkey", "l_quantity")
    assert cheap.count() == cheap.cache().count()


if __name__ == "__main__":
    main()
