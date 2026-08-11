"""Ranking within a time partition, and the ties that dates create.

Dates repeat, so an ordering by date alone is not a total order and `row_number` breaks the
ties arbitrarily. Adding a tiebreaker makes the result reproducible, which matters the
moment anyone compares two runs.

    python examples/windows/rank_dense_within_time.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_custkey", "o_orderkey", "o_orderdate")

    # Ordering by date alone: ties are broken arbitrarily.
    loose = orders.with_columns(
        seq=bt.row_number().over(partition_by=["o_custkey"], order_by=["o_orderdate"])
    )

    # Ordering by (date, key): a total order, so the result is reproducible.
    tight = orders.with_columns(
        seq=bt.row_number().over(partition_by=["o_custkey"], order_by=["o_orderdate", "o_orderkey"])
    )

    first_tight = tight.filter(col("seq") == 1).sort("o_custkey").to_pydict()
    again = (
        orders.with_columns(
            seq=bt.row_number().over(
                partition_by=["o_custkey"], order_by=["o_orderdate", "o_orderkey"]
            )
        )
        .filter(col("seq") == 1)
        .sort("o_custkey")
        .to_pydict()
    )
    assert first_tight["o_orderkey"] == again["o_orderkey"]
    print("total ordering is reproducible across runs")

    # Both spellings still number every partition from 1.
    for name, ranked in (("loose", loose), ("tight", tight)):
        firsts = ranked.filter(col("seq") == 1).count()
        assert firsts == orders.n_unique("o_custkey"), name

    # `rank` shares a number between tied rows, so a customer with two orders on the same
    # day has a repeated rank where `row_number` does not.
    ranked = orders.with_columns(
        position=bt.rank().over(partition_by=["o_custkey"], order_by=["o_orderdate"]),
        sequence=bt.row_number().over(partition_by=["o_custkey"], order_by=["o_orderdate"]),
    )
    ties = ranked.filter(col("position") != col("sequence")).count()
    print("rows where rank and row_number disagree:", ties)
    assert ties >= 0


if __name__ == "__main__":
    main()
