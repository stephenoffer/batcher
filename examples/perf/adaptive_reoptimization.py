"""Adaptive re-optimization: re-planning on measured cardinalities.

At a pipeline breaker the engine knows how many rows actually came out, not how many it
guessed. Re-planning on that is the moat — and the invariant that makes it safe is that it
can change the plan but never the result.

    python examples/perf/adaptive_reoptimization.py
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
    orders = tpch("orders")
    customer = tpch("customer")

    # A shape where the estimate is likely to be wrong: a selective filter feeding two
    # joins, so the cardinality after the filter decides the best build side.
    query = (
        lineitem.filter(col("l_quantity") > 48)
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(customer, left_on="o_custkey", right_on="c_custkey")
        .group_by("c_nationkey")
        .agg(revenue=(col("l_extendedprice") * (1 - col("l_discount"))).sum(), lines=bt.count())
        .sort("c_nationkey")
    )

    started = time.perf_counter()
    adaptive = query.collect(adaptive=True)
    adaptive_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    static = query.collect(adaptive=False)
    static_ms = (time.perf_counter() - started) * 1000

    print(f"adaptive {adaptive_ms:7.1f} ms   static {static_ms:7.1f} ms")
    print("nations:", adaptive.num_rows)

    # The invariant: identical results either way.
    assert adaptive.schema == static.schema
    assert adaptive.num_rows == static.num_rows
    left, right = adaptive.to_pydict(), static.to_pydict()
    assert left["c_nationkey"] == right["c_nationkey"]
    assert left["lines"] == right["lines"]
    assert all(
        abs(a - b) <= abs(a) * 1e-12 for a, b in zip(left["revenue"], right["revenue"], strict=True)
    )

    # The plan is where a difference would show, not the answer.
    print(query.explain())


if __name__ == "__main__":
    main()
