"""Controlling the column order of a result.

Column order is part of the output contract for anything that writes a CSV or feeds a
positional consumer. `select` fixes it explicitly; `with_columns` appends. Relying on the
order a join happens to produce is how a downstream reader silently shifts by one.

    python examples/relational/column_order_and_selection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")
    lineitem = tpch("lineitem")

    joined = orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
    print("join order:", joined.columns[:5], "...")

    # The join's order is left columns then right columns, which is defined but not
    # something to depend on when the shape of either side may change.
    assert joined.columns[: len(orders.columns)] == orders.columns

    # An explicit projection fixes the contract.
    contract = ["o_orderkey", "l_linenumber", "l_quantity", "o_totalprice"]
    fixed = joined.select(*contract)
    assert fixed.columns == contract

    # Appending never reorders what is already there.
    widened = fixed.with_columns(net=col("l_quantity") * col("o_totalprice"))
    assert widened.columns == [*contract, "net"]

    # Replacing a column keeps its position.
    replaced = widened.with_columns(l_quantity=col("l_quantity") * 2)
    assert replaced.columns == widened.columns

    # Dropping preserves the order of the survivors.
    trimmed = widened.drop("o_totalprice")
    assert trimmed.columns == ["o_orderkey", "l_linenumber", "l_quantity", "net"]

    # And a rename changes the label, not the position.
    renamed = trimmed.rename({"l_quantity": "units"})
    assert renamed.columns == ["o_orderkey", "l_linenumber", "units", "net"]


if __name__ == "__main__":
    main()
