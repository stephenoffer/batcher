"""Struct columns: packing several values into one, and reading them back.

A struct keeps related fields together through a pipeline without widening the schema, and
`unnest` flattens it again at the end. Reading one field is a projection, so unused fields
cost nothing.

    python examples/expr_collections/struct_columns.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = (
        tpch("orders")
        .select("o_orderkey", "o_totalprice", "o_orderstatus", "o_orderdate")
        .head(1_000)
    )

    packed = orders.select(
        "o_orderkey",
        detail=bt.struct(
            price=col("o_totalprice"),
            status=col("o_orderstatus"),
            day=col("o_orderdate"),
        ),
    )
    print(packed.head(2).to_pydict())
    assert packed.columns == ["o_orderkey", "detail"]

    # Reading one field back.
    prices = packed.select(price=col("detail").struct.field("price")).to_pydict()
    assert prices["price"] == orders.to_pydict()["o_totalprice"]

    # Filtering on a nested field works like any other expression.
    expensive = packed.filter(col("detail").struct.field("price") > 200_000)
    direct = orders.filter(col("o_totalprice") > 200_000)
    print("expensive orders:", expensive.count())
    assert expensive.count() == direct.count()

    # Flattening restores the flat schema, with the row count unchanged.
    flat = packed.unnest("detail")
    assert flat.count() == packed.count()
    assert {"price", "status", "day"} <= set(flat.columns)


if __name__ == "__main__":
    main()
