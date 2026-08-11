"""A full pipeline checked for single-node/distributed equivalence.

The contract is that the multiset of rows, every column name and every column type are exact,
and that floating-point reductions agree up to reassociation. This runs a realistic pipeline
and checks each of those separately, so a failure says which one broke.

    python examples/dist/end_to_end_distributed.py
    python examples/dist/end_to_end_distributed.py --distributed
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
    orders = tpch("orders")
    customer = tpch("customer")
    nation = tpch("nation")

    pipeline = (
        lineitem.filter(col("l_quantity") > 10)
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(customer, left_on="o_custkey", right_on="c_custkey")
        .join(nation, left_on="c_nationkey", right_on="n_nationkey")
        .with_columns(revenue=col("l_extendedprice") * (1 - col("l_discount")))
        .group_by("n_name", "l_shipmode")
        .agg(
            revenue=col("revenue").sum(),
            lines=bt.count(),
            biggest=col("revenue").max(),
            parts=bt.approx_n_unique(col("l_partkey")),
        )
        .sort("n_name", "l_shipmode")
    )

    single = pipeline.collect(distributed=False, num_partitions=1)
    many = pipeline.collect(distributed=distributed, num_partitions=8)
    print(f"{single.num_rows} nation/shipmode rows")

    # 1. Column names and types are exact.
    assert single.column_names == many.column_names
    assert single.schema == many.schema

    # 2. The row multiset is exact.
    assert single.num_rows == many.num_rows

    left, right = single.to_pydict(), many.to_pydict()
    assert left["n_name"] == right["n_name"]
    assert left["l_shipmode"] == right["l_shipmode"]

    # 3. Integer aggregates are exact.
    assert left["lines"] == right["lines"]
    assert left["parts"] == right["parts"]

    # 4. Floating-point reductions agree up to reassociation. `combine` is associative in
    # exact arithmetic and IEEE addition is not, so the partition count changes the
    # summation order; compensated summation bounds the difference to the last bits.
    assert all(
        abs(a - b) <= abs(a) * 1e-12
        for a, b in zip(left["revenue"], right["revenue"], strict=True)
    )
    # A max is order-independent, so it is exact.
    assert left["biggest"] == right["biggest"]

    print("names, types, rows, integers exact; floats agree to the last bits")


if __name__ == "__main__":
    main()
