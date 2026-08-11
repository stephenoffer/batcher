"""Keeping the most recent row per key, which `distinct` cannot do.

`drop_duplicates` keeps an arbitrary row per key. When "which one" matters — the latest
version, the highest score — rank within the key and keep rank 1. That is deterministic,
and it says in the query which row wins.

    python examples/relational/deduplicate_keeping_latest.py
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

    # The most recent order per customer, with ties broken by order key so the result is
    # deterministic rather than merely reproducible-looking.
    latest = (
        orders.with_columns(
            rank=bt.row_number().over(
                partition_by=["o_custkey"], order_by=[("o_orderdate", True), ("o_orderkey", True)]
            )
        )
        .filter(col("rank") == 1)
        .drop("rank")
    )

    print("customers:", latest.count())
    assert latest.count() == orders.n_unique("o_custkey")
    keys = latest.to_pydict()["o_custkey"]
    assert len(set(keys)) == len(keys)

    # The kept row really is the latest for its customer.
    sample_customer = keys[0]
    theirs = orders.filter(col("o_custkey") == sample_customer).to_pydict()
    kept = latest.filter(col("o_custkey") == sample_customer).to_pydict()
    assert kept["o_orderdate"][0] == max(theirs["o_orderdate"])

    # `drop_duplicates` gives one row per key too, but not necessarily that one.
    arbitrary = orders.drop_duplicates(subset=["o_custkey"])
    assert arbitrary.count() == latest.count()

    # Running the ranked version twice gives an identical answer.
    again = (
        orders.with_columns(
            rank=bt.row_number().over(
                partition_by=["o_custkey"], order_by=[("o_orderdate", True), ("o_orderkey", True)]
            )
        )
        .filter(col("rank") == 1)
        .drop("rank")
    )
    assert (
        again.sort("o_custkey").to_pydict()["o_orderkey"]
        == latest.sort("o_custkey").to_pydict()["o_orderkey"]
    )


if __name__ == "__main__":
    main()
