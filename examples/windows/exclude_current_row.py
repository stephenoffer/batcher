"""Leave-one-out aggregates: the group's total without this row.

"How does this row compare to its peers" needs the peers *excluding* the row itself,
otherwise the row is compared partly against itself. Subtracting the row from the partition
total is the cheap way; a split frame is the general one.

    python examples/windows/exclude_current_row.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_orderkey", "l_linenumber", "l_extendedprice").head(30_000)

    compared = (
        lineitem.with_columns(
            order_total=col("l_extendedprice").sum().over(partition_by=["l_orderkey"]),
            # Count the column rather than the rows: `bt.count()` is the aggregate form
            # and does not carry a window.
            order_lines=col("l_extendedprice").count().over(partition_by=["l_orderkey"]),
        )
        .with_columns(
            others_total=col("order_total") - col("l_extendedprice"),
            others_count=col("order_lines") - 1,
        )
        .with_columns(
            others_mean=bt.when(col("others_count") > 0)
            .then(col("others_total") / col("others_count"))
            .otherwise(0.0)
        )
    )

    result = compared.sort("l_orderkey", "l_linenumber").head(6).to_pydict()
    for row in zip(
        result["l_orderkey"], result["l_extendedprice"], result["others_mean"], strict=True
    ):
        print(f"  order {row[0]:>8} line {row[1]:>10,.2f} peers mean {row[2]:>10,.2f}")

    full = compared.to_pydict()

    # The leave-one-out total plus the row is the partition total.
    assert all(
        abs(others + value - total) < 1e-6
        for others, value, total in zip(
            full["others_total"], full["l_extendedprice"], full["order_total"], strict=True
        )
    )

    # A single-line order has no peers, and the guard keeps it from dividing by zero.
    singles = compared.filter(col("others_count") == 0).to_pydict()
    assert all(value == 0.0 for value in singles["others_mean"])
    assert all(abs(value) < 1e-6 for value in singles["others_total"])

    # Rows above their peers, which is the question this was all for.
    above = compared.filter(
        (col("others_count") > 0) & (col("l_extendedprice") > col("others_mean"))
    )
    print("lines above their order's other lines:", above.count())
    assert 0 < above.count() < lineitem.count()


if __name__ == "__main__":
    main()
