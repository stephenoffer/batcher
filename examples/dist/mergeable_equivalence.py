"""The contract that makes distribution safe: partial, combine, finalize.

A stateful operator is built as three mergeable pieces, so running it on one core and on a
cluster is the same algebra with a different schedule. This example asserts the property
directly: the same query over 1 partition and over 8 gives the same answer.

Run it against a real cluster with `--distributed`; without that it exercises the same
mergeable path over several local partitions, which is what the property is actually about.

    python examples/dist/mergeable_equivalence.py
    python examples/dist/mergeable_equivalence.py --distributed
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
    print("distributed:", distributed)

    lineitem = tpch("lineitem")

    query = (
        lineitem.group_by("l_returnflag", "l_linestatus")
        .agg(
            lines=bt.count(),
            qty=col("l_quantity").sum(),
            revenue=(col("l_extendedprice") * (1 - col("l_discount"))).sum(),
            distinct_parts=bt.approx_n_unique(col("l_partkey")),
        )
        .sort("l_returnflag", "l_linestatus")
    )

    single = query.collect(distributed=False, num_partitions=1)
    many = query.collect(distributed=distributed, num_partitions=8)

    print(single.to_pydict()["lines"])

    # Row multiset, column names and column types are exact.
    assert single.schema == many.schema
    assert single.num_rows == many.num_rows

    left = single.to_pydict()
    right = many.to_pydict()
    assert left["l_returnflag"] == right["l_returnflag"]
    assert left["lines"] == right["lines"]
    assert left["qty"] == right["qty"]

    # Floating-point reductions are identical only up to reassociation: `combine` is
    # associative in exact arithmetic and IEEE addition is not, so the partition count
    # changes the summation order. Compensated summation bounds that to the last bits.
    assert all(
        abs(a - b) <= abs(a) * 1e-12
        for a, b in zip(left["revenue"], right["revenue"], strict=True)
    )

    # A mergeable sketch combines across partitions to the same estimate.
    assert left["distinct_parts"] == right["distinct_parts"]


if __name__ == "__main__":
    main()
