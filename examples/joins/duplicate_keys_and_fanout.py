"""Fan-out: what a non-unique join key does to your row count.

An inner join emits one row per matching *pair*. If the right side has three rows for a
key, one left row becomes three. That is correct join semantics and almost never what
someone means when they say "look up the price" — so check the key's uniqueness first.

    python examples/joins/duplicate_keys_and_fanout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_totalprice")
    lineitem = tpch("lineitem").select("l_orderkey", "l_extendedprice")

    # Is the right-hand key unique? One count answers it.
    right_rows = lineitem.count()
    right_keys = lineitem.n_unique("l_orderkey")
    print(f"lineitem: {right_rows} rows, {right_keys} distinct order keys")
    assert right_keys < right_rows  # not unique: expect fan-out

    joined = orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
    print("joined rows:", joined.count())

    # The fan-out is exactly the sum of the right-hand group sizes over matched keys.
    matched = lineitem.group_by("l_orderkey").agg(n=bt.count())
    expected = (
        orders.join(matched, left_on="o_orderkey", right_on="l_orderkey")
        .agg(total=col("n").sum())
        .to_pydict()["total"][0]
    )
    assert joined.count() == expected

    # Pre-aggregating the right side makes the key unique and the join safe.
    reduced = lineitem.group_by("l_orderkey").agg(line_total=col("l_extendedprice").sum())
    assert reduced.count() == reduced.n_unique("l_orderkey")
    safe = orders.join(reduced, left_on="o_orderkey", right_on="l_orderkey")
    assert safe.count() <= orders.count()


if __name__ == "__main__":
    main()
