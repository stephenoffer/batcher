"""Filtering: combining predicates, and what nulls do to them.

`&` and `|` need parentheses because Python binds them tighter than comparison. The
subtler point is three-valued logic: a comparison against null is null, not false, so a
row with a null key survives neither `x == 1` nor `x != 1`.

    python examples/relational/filter_predicates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")
    total = orders.count()

    # Conjunction. The parentheses are mandatory.
    expensive_open = orders.filter((col("o_totalprice") > 100_000) & (col("o_orderstatus") == "O"))
    print("expensive open orders:", expensive_open.count())

    # Chained `.filter` calls mean the same thing and read better when the predicates
    # are unrelated.
    chained = orders.filter(col("o_totalprice") > 100_000).filter(col("o_orderstatus") == "O")
    assert chained.count() == expensive_open.count()

    # Disjunction, negation, and set membership.
    urgent = orders.filter(col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"]))
    not_urgent = orders.filter(~col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"]))
    print("urgent:", urgent.count(), "other:", not_urgent.count())

    # No order has a null priority, so the two halves partition the table exactly. If any
    # were null this identity would fail — which is the point of checking it.
    assert urgent.count() + not_urgent.count() == total
    assert orders.filter(col("o_orderpriority").is_null()).count() == 0

    # Comparing columns to each other, not just to constants.
    lineitem = tpch("lineitem")
    late = lineitem.filter(col("l_receiptdate") > col("l_commitdate"))
    print("late lines:", late.count())
    assert 0 < late.count() < lineitem.count()


if __name__ == "__main__":
    main()
