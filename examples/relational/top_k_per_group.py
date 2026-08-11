"""The top N rows within each group, two ways.

A global top-N is `top_k`. A *per-group* top-N needs a ranking window, then a filter on
the rank. The window version is the one that generalizes, and it is what to reach for
when someone asks for "the three biggest orders per customer".

    python examples/relational/top_k_per_group.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_custkey", "o_orderkey", "o_totalprice")

    # Global: the ten largest orders overall.
    biggest = orders.top_k(10, by="o_totalprice").to_pydict()
    print("largest:", [round(value) for value in biggest["o_totalprice"][:5]])
    assert biggest["o_totalprice"] == sorted(biggest["o_totalprice"], reverse=True)

    # Per group: rank inside each customer, then keep the top three.
    ranked = orders.with_columns(
        # `order_by` takes (column, descending) pairs, so the ranking direction is
        # part of the window rather than a separate argument.
        rank=bt.row_number().over(partition_by=["o_custkey"], order_by=[("o_totalprice", True)])
    )
    top_three = ranked.filter(col("rank") <= 3)

    counts = top_three.group_by("o_custkey").agg(kept=bt.count()).to_pydict()
    assert all(kept <= 3 for kept in counts["kept"])

    # And the ranking really is by descending price inside a group.
    sample_customer = counts["o_custkey"][0]
    rows = top_three.filter(col("o_custkey") == sample_customer).sort("rank").to_pydict()
    print("customer", sample_customer, [round(value) for value in rows["o_totalprice"]])
    assert rows["o_totalprice"] == sorted(rows["o_totalprice"], reverse=True)
    assert rows["rank"] == list(range(1, len(rows["rank"]) + 1))


if __name__ == "__main__":
    main()
