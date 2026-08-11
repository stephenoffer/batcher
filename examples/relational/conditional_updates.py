"""Updating a column in place, conditionally.

There is no `UPDATE` here — a Dataset is immutable. The equivalent is `with_columns` with a
CASE that leaves the untouched rows alone, which is also safer: the old value is still
available in the same expression.

    python examples/relational/conditional_updates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_orderstatus", "o_totalprice")

    # "Apply a 10% surcharge to open orders" — everything else is untouched.
    adjusted = orders.with_columns(
        o_totalprice=bt.when(col("o_orderstatus") == "O")
        .then(col("o_totalprice") * 1.1)
        .otherwise(col("o_totalprice"))
    )

    before = orders.sort("o_orderkey").to_pydict()
    after = adjusted.sort("o_orderkey").to_pydict()

    changed = sum(
        1
        for old, new in zip(before["o_totalprice"], after["o_totalprice"], strict=True)
        if old != new
    )
    open_orders = orders.filter(col("o_orderstatus") == "O").count()
    print(f"{changed} of {orders.count()} rows changed")
    assert changed == open_orders

    # Untouched rows are bit-identical, not merely close.
    assert all(
        new == old
        for status, old, new in zip(
            before["o_orderstatus"], before["o_totalprice"], after["o_totalprice"], strict=True
        )
        if status != "O"
    )

    # The schema does not move: same columns, same order, same types.
    assert adjusted.columns == orders.columns
    assert adjusted.dtypes == orders.dtypes


if __name__ == "__main__":
    main()
