"""Aggregates that depend on order, and how to make them deterministic.

`first`, `last` and `array_agg` all depend on which row came first, so the answer is only
reproducible if the order is total. Supplying the ordering explicitly is the difference
between a result you can compare across runs and one you cannot.

    python examples/aggregations/ordered_aggregates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_custkey", "o_orderkey", "o_orderdate", "o_totalprice")

    # An explicit ordering column makes `first` and `last` well defined.
    per_customer = (
        orders.group_by("o_custkey")
        .agg(
            first_order=bt.first(col("o_orderkey"), col("o_orderdate")),
            last_order=bt.last(col("o_orderkey"), col("o_orderdate")),
            earliest=bt.first(col("o_orderdate"), col("o_orderdate")),
            latest=bt.last(col("o_orderdate"), col("o_orderdate")),
            orders=bt.count(),
        )
        .sort("o_custkey")
    )
    result = per_customer.to_pydict()
    print(per_customer.head(3).to_pydict())

    # Ordered by date, so first is not after last.
    assert all(
        early <= late for early, late in zip(result["earliest"], result["latest"], strict=True)
    )

    # Running it again gives the same answer.
    again = (
        orders.group_by("o_custkey")
        .agg(first_order=bt.first(col("o_orderkey"), col("o_orderdate")))
        .sort("o_custkey")
        .to_pydict()
    )
    assert again["first_order"] == result["first_order"]

    # A single-order customer has the same first and last.
    singles = per_customer.filter(col("orders") == 1).to_pydict()
    assert all(
        first == last
        for first, last in zip(singles["first_order"], singles["last_order"], strict=True)
    )

    # `array_agg` collects in arrival order, so sort the input when the order matters.
    collected = (
        orders.sort("o_custkey", "o_orderdate")
        .group_by("o_custkey")
        .agg(dates=bt.array_agg(col("o_orderdate")))
        .sort("o_custkey")
        .head(5)
        .to_pydict()
    )
    print(collected["dates"][0])
    assert all(len(value) >= 1 for value in collected["dates"])


if __name__ == "__main__":
    main()
