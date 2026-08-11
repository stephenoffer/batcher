"""Filtering joins: semi keeps matches, anti keeps orphans.

Both return only left-hand columns and neither can change the row count upward — that is
what makes them filters rather than joins. Using an inner join for a membership test is
the bug they exist to prevent: it multiplies rows whenever the right side has duplicates.

    python examples/joins/semi_and_anti.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_custkey", "o_totalprice")
    lineitem = tpch("lineitem").select("l_orderkey", "l_quantity")

    with_lines = orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey", how="semi")
    without_lines = orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey", how="anti")

    print("orders with lines:", with_lines.count(), "without:", without_lines.count())

    # The two partition the left side exactly.
    assert with_lines.count() + without_lines.count() == orders.count()

    # Neither adds columns from the right.
    assert with_lines.columns == orders.columns
    assert without_lines.columns == orders.columns

    # The inner join is the one that multiplies: each order appears once per line.
    inner = orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
    print("inner rows:", inner.count())
    assert inner.count() > with_lines.count()

    # An anti join is the orphan check: these order keys really are absent downstream.
    missing = without_lines.select("o_orderkey").head(5).to_pydict()["o_orderkey"]
    for key in missing:
        assert lineitem.filter(col("l_orderkey") == key).count() == 0


if __name__ == "__main__":
    main()
