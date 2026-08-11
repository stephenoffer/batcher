"""Positional and most-common aggregates: first, last, and mode.

`first` and `last` take the ordering column as a required argument, which is the API
refusing to let you ask an ill-defined question: "the first row" means nothing until
you say first *by what*. `mode` needs no order — it is the most frequent value.

    python examples/aggregations/first_last_and_mode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_custkey", "o_orderkey", "o_orderdate", "o_orderpriority")

    # The ordering column is part of the aggregate, so no prior sort is needed.
    per_customer = (
        orders.group_by("o_custkey")
        .agg(
            earliest=bt.first(col("o_orderdate"), col("o_orderdate")),
            latest=bt.last(col("o_orderdate"), col("o_orderdate")),
            first_priority=bt.first(col("o_orderpriority"), col("o_orderdate")),
            orders=bt.count(),
        )
        .sort("o_custkey")
        .head(5)
        .to_pydict()
    )
    print(per_customer)

    # Ordered by date, so the earliest cannot be later than the latest.
    assert all(
        early <= late
        for early, late in zip(per_customer["earliest"], per_customer["latest"], strict=True)
    )

    # The most frequent value in a column.
    common = orders.agg(usual_priority=bt.mode(col("o_orderpriority"))).to_pydict()
    print("most common priority:", common["usual_priority"][0])

    # Cross-check: the mode really is the value with the largest count.
    counts = orders.value_counts("o_orderpriority").sort("count", descending=True).to_pydict()
    assert common["usual_priority"][0] == counts["o_orderpriority"][0]


if __name__ == "__main__":
    main()
