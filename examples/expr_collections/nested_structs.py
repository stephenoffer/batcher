"""Structs inside structs, and reaching into them.

Nesting keeps a hierarchy intact through a pipeline instead of flattening it into a wide
schema with prefixed names. Reading a leaf is a chain of `.struct.field`, and it is still a
projection — the untouched branches cost nothing.

    python examples/expr_collections/nested_structs.py
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
        .select("o_orderkey", "o_custkey", "o_totalprice", "o_orderstatus", "o_orderdate")
        .head(1_000)
    )

    nested = orders.select(
        "o_orderkey",
        detail=bt.struct(
            customer=bt.struct(key=col("o_custkey")),
            money=bt.struct(total=col("o_totalprice"), status=col("o_orderstatus")),
            placed=col("o_orderdate"),
        ),
    )
    print(nested.head(1).to_pydict())
    assert nested.columns == ["o_orderkey", "detail"]

    # Reaching a leaf, two levels down.
    leaves = nested.select(
        "o_orderkey",
        customer=col("detail").struct.field("customer").struct.field("key"),
        total=col("detail").struct.field("money").struct.field("total"),
    )
    values = leaves.to_pydict()
    original = orders.to_pydict()
    assert values["customer"] == original["o_custkey"]
    assert values["total"] == original["o_totalprice"]

    # Filtering on a nested leaf.
    expensive = nested.filter(col("detail").struct.field("money").struct.field("total") > 200_000)
    direct = orders.filter(col("o_totalprice") > 200_000)
    print("expensive:", expensive.count())
    assert expensive.count() == direct.count()

    # Flattening one level at a time.
    once = nested.unnest("detail")
    assert {"customer", "money", "placed"} <= set(once.columns)
    twice = once.unnest("money")
    assert {"total", "status"} <= set(twice.columns)
    assert twice.count() == orders.count()


if __name__ == "__main__":
    main()
