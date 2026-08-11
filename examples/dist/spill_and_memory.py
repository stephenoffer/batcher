"""Bounded memory: what spilling buys and what it costs.

Spilling is what keeps a query alive when its state does not fit. It is also the mechanism
that makes the distributed path safe: per-node memory stays bounded because each node's
partial state can go to disk. The result is identical either way.

    python examples/dist/spill_and_memory.py
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

    # High-cardinality state: one group per order key.
    query = (
        lineitem.group_by("l_orderkey")
        .agg(revenue=col("l_extendedprice").sum(), lines=bt.count())
        .sort("l_orderkey")
    )

    started = time.perf_counter()
    in_memory = query.collect()
    memory_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    spilled = query.collect(spill=True)
    spill_ms = (time.perf_counter() - started) * 1000

    print(f"groups: {in_memory.num_rows}")
    print(f"in memory {memory_ms:7.1f} ms   spilled {spill_ms:7.1f} ms")

    # Identical results, which is what makes spilling safe to leave enabled.
    assert spilled.schema == in_memory.schema
    assert spilled.num_rows == in_memory.num_rows
    left, right = in_memory.to_pydict(), spilled.to_pydict()
    assert left["l_orderkey"] == right["l_orderkey"]
    assert left["lines"] == right["lines"]
    assert all(
        abs(a - b) < 1e-6 for a, b in zip(left["revenue"], right["revenue"], strict=True)
    )

    # A sort spills too, and its order must survive.
    top = lineitem.sort("l_extendedprice", descending=True).head(20)
    assert (
        top.collect(spill=True).to_pydict()["l_extendedprice"]
        == top.collect().to_pydict()["l_extendedprice"]
    )


if __name__ == "__main__":
    main()
