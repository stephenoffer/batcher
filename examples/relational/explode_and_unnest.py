"""Nested data: exploding a list column and flattening a struct.

`explode` turns one row with an n-element list into n rows, dropping rows whose list is
empty. `unnest` is the struct version and changes no row count at all — it only lifts
fields into top-level columns.

    python examples/relational/explode_and_unnest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").head(500)

    # Build a list column from real data: the words of each order's clerk field.
    with_parts = orders.select(
        "o_orderkey",
        parts=col("o_clerk").str.split("#"),
    )
    print(with_parts.head(2).to_pydict())

    exploded = with_parts.explode("parts")
    # Every clerk id is "Clerk#000000xxx", so each row becomes exactly two.
    assert exploded.count() == with_parts.count() * 2

    # A struct column, then flattened. The row count does not move.
    packed = orders.select(
        "o_orderkey",
        # `struct` names its fields with keywords — the name is the keyword, so there
        # is no separate alias step.
        detail=bt.struct(price=col("o_totalprice"), status=col("o_orderstatus")),
    )
    flat = packed.unnest("detail")
    print(flat.head(2).to_pydict())
    assert flat.count() == packed.count()
    assert {"price", "status"} <= set(flat.columns)

    # Reading a field without flattening.
    field = packed.select(price=col("detail").struct.field("price")).head(3).to_pydict()
    assert field["price"] == orders.select("o_totalprice").head(3).to_pydict()["o_totalprice"]


if __name__ == "__main__":
    main()
