"""Set operations: union, intersect, and except, with and without duplicates.

The three set operators require both sides to agree on column names and types. `union`
concatenates and keeps duplicates; putting `.distinct()` after it is the SQL `UNION`
rather than `UNION ALL`.

    python examples/relational/set_operations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    customer = tpch("customer")

    rich = customer.filter(col("c_acctbal") > 8000).select("c_custkey")
    building = customer.filter(col("c_mktsegment") == "BUILDING").select("c_custkey")

    print("rich:", rich.count(), "building:", building.count())

    # UNION ALL: every row from both sides, duplicates included.
    both = rich.union(building)
    assert both.count() == rich.count() + building.count()

    # UNION: the distinct set.
    either = both.distinct()
    assert either.count() <= both.count()

    # INTERSECT and EXCEPT.
    overlap = rich.intersect(building)
    only_rich = rich.except_(building)
    print("both:", overlap.count(), "rich only:", only_rich.count())

    # The three counts have to reconcile: |A| = |A ∩ B| + |A \ B|.
    assert overlap.count() + only_rich.count() == rich.distinct().count()

    # `vstack`/`append` are the positional concatenation, for when the two sides are
    # already known to line up.
    stacked = rich.head(10).vstack(building.head(10))
    assert stacked.count() == 20


if __name__ == "__main__":
    main()
