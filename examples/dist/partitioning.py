"""Partition count: what it changes, and what it must not.

Partitioning decides how work is divided and therefore how much parallelism and how much
per-node memory a query uses. It must not decide the answer. Sweeping the partition count
and asserting the result is stable is the cheapest way to catch an operator that is not
properly mergeable.

    python examples/dist/partitioning.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import resolve_distributed, tpch
from batcher import col


def main() -> None:
    distributed = resolve_distributed()
    lineitem = tpch("lineitem")

    query = (
        lineitem.group_by("l_shipmode")
        .agg(lines=bt.count(), qty=col("l_quantity").sum(), biggest=col("l_quantity").max())
        .sort("l_shipmode")
    )

    baseline = query.collect(distributed=False, num_partitions=1).to_pydict()
    print("baseline:", baseline["lines"])

    for partitions in (2, 4, 8, 16):
        result = query.collect(distributed=distributed, num_partitions=partitions).to_pydict()
        assert result["l_shipmode"] == baseline["l_shipmode"], partitions
        assert result["lines"] == baseline["lines"], partitions
        assert result["qty"] == baseline["qty"], partitions
        assert result["biggest"] == baseline["biggest"], partitions
    print("stable across 1, 2, 4, 8 and 16 partitions")

    # `repartition` changes the layout of a Dataset without changing its contents.
    spread = lineitem.repartition(8)
    assert spread.count() == lineitem.count()

    # An order-sensitive result needs the sort to be part of the query, not an accident
    # of the partitioning — this is the case that silently breaks at scale.
    top = lineitem.sort("l_extendedprice", descending=True).head(10)
    assert (
        top.collect(num_partitions=1).to_pydict()["l_extendedprice"]
        == top.collect(num_partitions=8).to_pydict()["l_extendedprice"]
    )


if __name__ == "__main__":
    main()
