"""Grouping by several columns, and by an expression.

The group key can be any expression, not only a column, so "revenue by year" needs no
materialized year column. The key's *cardinality* is what decides the cost: a two-column
key is not twice the work of one, it is as much work as the number of distinct pairs.

    python examples/aggregations/multi_key_grouping.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")

    # A derived key: group by the year of a date column.
    by_year = (
        orders.with_columns(year=col("o_orderdate").dt.year())
        .group_by("year")
        .agg(orders=bt.count(), revenue=col("o_totalprice").sum())
        .sort("year")
        .to_pydict()
    )
    print(by_year["year"], by_year["orders"])
    assert by_year["year"] == sorted(by_year["year"])
    assert sum(by_year["orders"]) == orders.count()

    # Two keys: the number of groups is the number of distinct *pairs*, which is at most
    # the product but usually far less.
    by_year_and_status = (
        orders.with_columns(year=col("o_orderdate").dt.year())
        .group_by("year", "o_orderstatus")
        .agg(orders=bt.count())
        .to_pydict()
    )
    years = len(by_year["year"])
    statuses = orders.n_unique("o_orderstatus")
    print(f"{len(by_year_and_status['orders'])} pairs, at most {years * statuses}")
    assert len(by_year_and_status["orders"]) <= years * statuses
    assert sum(by_year_and_status["orders"]) == orders.count()

    # Rolling the finer grouping up must reproduce the coarser one.
    rolled: dict[int, int] = {}
    for year, count in zip(by_year_and_status["year"], by_year_and_status["orders"], strict=True):
        rolled[year] = rolled.get(year, 0) + count
    assert rolled == dict(zip(by_year["year"], by_year["orders"], strict=True))


if __name__ == "__main__":
    main()
