"""Joining on more than one column.

A composite key is a list on each side, and the two lists have to line up positionally.
`partsupp` is the natural case: it is keyed by part *and* supplier, and joining on either
alone silently produces a cross product within the key.

    python examples/joins/multi_key_joins.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch


def main() -> None:
    lineitem = tpch("lineitem").select("l_orderkey", "l_partkey", "l_suppkey", "l_quantity")
    partsupp = tpch("partsupp").select("ps_partkey", "ps_suppkey", "ps_supplycost")

    # Both key columns, in matching order.
    correct = lineitem.join(
        partsupp,
        left_on=["l_partkey", "l_suppkey"],
        right_on=["ps_partkey", "ps_suppkey"],
    )
    print("composite join rows:", correct.count())

    # One key only: every line now matches every supplier of that part.
    partial = lineitem.join(partsupp, left_on="l_partkey", right_on="ps_partkey")
    print("partial-key join rows:", partial.count())

    # The composite join cannot fan out, because (part, supplier) is unique in partsupp.
    assert correct.count() <= lineitem.count()
    assert partial.count() > correct.count()

    # Same-named keys on both sides collapse to `on=[...]`.
    renamed = partsupp.rename({"ps_partkey": "l_partkey", "ps_suppkey": "l_suppkey"})
    shorthand = lineitem.join(renamed, on=["l_partkey", "l_suppkey"])
    assert shorthand.count() == correct.count()


if __name__ == "__main__":
    main()
