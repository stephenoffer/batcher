"""Repartitioning: changing the parallelism without changing the data.

Too few partitions and cores sit idle; too many and the per-partition overhead dominates.
The right number depends on the machine, which is why it is a knob rather than a constant —
and why the only invariant worth asserting is that it never changes the answer.

    python examples/perf/repartitioning.py
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

    query = (
        lineitem.group_by("l_shipmode", "l_returnflag")
        .agg(lines=bt.count(), revenue=col("l_extendedprice").sum())
        .sort("l_shipmode", "l_returnflag")
    )

    baseline = query.collect(num_partitions=1).to_pydict()
    print("groups:", len(baseline["l_shipmode"]))

    for partitions in (1, 2, 4, 8, 16, 32):
        started = time.perf_counter()
        result = query.collect(num_partitions=partitions).to_pydict()
        elapsed = (time.perf_counter() - started) * 1000
        print(f"  {partitions:>3} partitions  {elapsed:7.1f} ms")

        assert result["l_shipmode"] == baseline["l_shipmode"], partitions
        assert result["lines"] == baseline["lines"], partitions
        assert all(
            abs(a - b) <= abs(a) * 1e-12
            for a, b in zip(baseline["revenue"], result["revenue"], strict=True)
        ), partitions

    # `repartition` changes the layout of a Dataset without changing its contents.
    spread = lineitem.repartition(16)
    assert spread.count() == lineitem.count()
    assert (
        spread.agg(t=col("l_quantity").sum()).to_pydict()["t"]
        == (lineitem.agg(t=col("l_quantity").sum()).to_pydict()["t"])
    )

    # `shuffle` reorders rows without losing any, which is the other layout knob.
    mixed = lineitem.select("l_orderkey").shuffle(seed=3)
    assert mixed.count() == lineitem.count()
    assert bt is not None


if __name__ == "__main__":
    main()
