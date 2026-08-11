"""What a distributed join costs: the shuffle.

Joining two large tables means moving rows so that matching keys land on the same worker.
That movement is the expensive part, and it is why a broadcast — sending the small side
everywhere instead — wins whenever one side is small enough to fit.

    python examples/dist/shuffle_and_joins.py
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
    orders = tpch("orders")
    nation = tpch("nation")

    # Big-to-big: both sides must be shuffled by the join key.
    big_join = (
        lineitem.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .group_by("o_orderstatus")
        .agg(revenue=(col("l_extendedprice") * (1 - col("l_discount"))).sum())
        .sort("o_orderstatus")
    )

    # Big-to-tiny: 25 rows can be sent everywhere, so no shuffle of the fact table.
    customer = tpch("customer")
    broadcast_join = (
        customer.join(nation, left_on="c_nationkey", right_on="n_nationkey")
        .group_by("n_name")
        .agg(customers=bt.count())
        .sort("n_name")
    )

    for name, query in (("shuffle join", big_join), ("broadcast join", broadcast_join)):
        single = query.collect(distributed=False, num_partitions=1)
        many = query.collect(distributed=distributed, num_partitions=8)
        print(f"{name}: {single.num_rows} rows")
        assert single.schema == many.schema
        assert single.num_rows == many.num_rows
        left, right = single.to_pydict(), many.to_pydict()
        for column, values in left.items():
            if isinstance(values[0], float):
                assert all(
                    abs(a - b) <= abs(a) * 1e-12
                    for a, b in zip(values, right[column], strict=True)
                ), column
            else:
                assert values == right[column], column

    # The plan shows the join; how it is executed is the scheduler's decision.
    print(big_join.explain())


if __name__ == "__main__":
    main()
