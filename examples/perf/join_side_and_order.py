"""Which side of a join is built, and why it matters.

A hash join builds a table from one side and probes it with the other. Building from the
smaller side is what keeps the hash table in cache. The optimizer picks, but the shape of
the query — especially where the filters are — is what gives it something to pick from.

    python examples/perf/join_side_and_order.py
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

    # Filter first: the build side shrinks to a fraction of the table.
    def timed(label: str, build) -> None:
        started = time.perf_counter()
        rows = build().count()
        print(f"{label:<34} {rows:>8} rows  {(time.perf_counter() - started) * 1000:7.1f} ms")

    filtered_first = lambda: lineitem.join(  # noqa: E731
        orders.filter(col("o_orderstatus") == "F"),
        left_on="l_orderkey",
        right_on="o_orderkey",
    )
    filtered_after = lambda: lineitem.join(  # noqa: E731
        orders, left_on="l_orderkey", right_on="o_orderkey"
    ).filter(col("o_orderstatus") == "F")

    timed("filter the build side first", filtered_first)
    timed("filter after the join", filtered_after)

    # Both orderings are the same query, so they must agree exactly.
    assert filtered_first().count() == filtered_after().count()

    left = filtered_first().agg(t=col("l_extendedprice").sum()).to_pydict()["t"][0]
    right = filtered_after().agg(t=col("l_extendedprice").sum()).to_pydict()["t"][0]
    assert abs(left - right) < 1e-3

    # The plan shows where the filter ended up.
    print(filtered_after().explain())
    assert bt is not None


if __name__ == "__main__":
    main()
