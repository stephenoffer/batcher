"""How a query's cost moves as the partition count changes.

More partitions means more parallelism and more coordination. The curve has a minimum, and it
is a property of the machine and the query rather than a constant — which is why the only
thing worth asserting is that the answer does not move along it.

    python examples/dist/scaling_characteristics.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import resolve_distributed, tpch
from batcher import col


def main() -> None:
    distributed = resolve_distributed()
    lineitem = tpch("lineitem")

    # Two shapes with different coordination costs.
    low_cardinality = (
        lineitem.group_by("l_returnflag").agg(n=bt.count()).sort("l_returnflag")
    )
    high_cardinality = (
        lineitem.group_by("l_orderkey")
        .agg(revenue=col("l_extendedprice").sum())
        .sort("l_orderkey")
    )

    for name, query in (("few groups", low_cardinality), ("many groups", high_cardinality)):
        baseline = query.collect(distributed=False, num_partitions=1).to_pydict()
        print(f"{name} ({len(baseline[next(iter(baseline))])} rows)")

        for partitions in (1, 4, 16):
            started = time.perf_counter()
            result = query.collect(
                distributed=distributed, num_partitions=partitions
            ).to_pydict()
            elapsed = (time.perf_counter() - started) * 1000
            print(f"    {partitions:>3} partitions  {elapsed:7.1f} ms")

            for column, values in baseline.items():
                if values and isinstance(values[0], float):
                    assert all(
                        abs(a - b) <= max(abs(a), 1.0) * 1e-12
                        for a, b in zip(values, result[column], strict=True)
                    ), (name, partitions, column)
                else:
                    assert values == result[column], (name, partitions, column)

    # The two shapes differ in how much has to move, which is the thing that decides how
    # the curve looks.
    assert high_cardinality.count() > low_cardinality.count()
    print(
        f"{high_cardinality.count()} groups shuffle far more than "
        f"{low_cardinality.count()}"
    )


if __name__ == "__main__":
    main()
