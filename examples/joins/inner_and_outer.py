"""Inner, left, right and full outer over real tables.

The join type decides what happens to rows with no partner. Assert the row count against
something you can derive independently, because a join that silently drops half its input
still returns plausible-looking data.

    python examples/joins/inner_and_outer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    customer = tpch("customer").select("c_custkey", "c_name", "c_nationkey")
    nation = tpch("nation").select("n_nationkey", "n_name")

    # Every customer has a nation, so inner and left agree here.
    inner = customer.join(nation, left_on="c_nationkey", right_on="n_nationkey")
    left = customer.join(nation, left_on="c_nationkey", right_on="n_nationkey", how="left")
    assert inner.count() == left.count() == customer.count()

    # Drop some nations and the two diverge: left keeps the orphans with nulls.
    few_nations = nation.filter(col("n_nationkey") < 5)
    inner_few = customer.join(few_nations, left_on="c_nationkey", right_on="n_nationkey")
    left_few = customer.join(few_nations, left_on="c_nationkey", right_on="n_nationkey", how="left")
    print("inner:", inner_few.count(), "left:", left_few.count())
    assert left_few.count() == customer.count()
    assert inner_few.count() < left_few.count()

    # The rows left adds are exactly the ones with a null right-hand column.
    orphans = left_few.filter(col("n_name").is_null()).count()
    assert inner_few.count() + orphans == left_few.count()

    # Right is the mirror image: every nation survives, customers may be null.
    right = few_nations.join(customer, left_on="n_nationkey", right_on="c_nationkey", how="right")
    assert right.count() == customer.count()

    # Full outer keeps both sides' unmatched rows.
    outer = customer.join(few_nations, left_on="c_nationkey", right_on="n_nationkey", how="outer")
    assert outer.count() >= left_few.count()


if __name__ == "__main__":
    main()
