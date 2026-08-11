"""Reconciling two datasets: what is in one and not the other, both ways.

The two anti joins plus the intersection partition the union, which makes them a complete
answer rather than a spot check. This is the shape of every "did the migration lose rows"
investigation.

    python examples/relational/anti_join_reconciliation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_totalprice")

    # Two overlapping extracts of the same table, as two systems would produce.
    left = orders.filter(col("o_orderkey") % 3 != 0).select("o_orderkey")
    right = orders.filter(col("o_orderkey") % 5 != 0).select("o_orderkey")

    only_left = left.join(right, on="o_orderkey", how="anti")
    only_right = right.join(left, on="o_orderkey", how="anti")
    both = left.join(right, on="o_orderkey", how="semi")

    print(f"left {left.count()}  right {right.count()}")
    print(f"only left {only_left.count()}  only right {only_right.count()}  both {both.count()}")

    # The three parts partition each side exactly.
    assert only_left.count() + both.count() == left.count()
    assert only_right.count() + both.count() == right.count()

    # And they reconstruct the union with nothing double-counted.
    union = left.union(right).distinct()
    assert only_left.count() + only_right.count() + both.count() == union.count()

    # A spot check on one difference, so the counts are not just internally consistent.
    sample = only_left.head(1).to_pydict()["o_orderkey"][0]
    assert right.filter(col("o_orderkey") == sample).count() == 0
    assert left.filter(col("o_orderkey") == sample).count() == 1


if __name__ == "__main__":
    main()
