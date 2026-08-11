"""GROUP BY and HAVING in SQL, and the DataFrame equivalent.

`HAVING` filters groups and `WHERE` filters rows, and they are not interchangeable. Writing
both in one query and comparing the counts is the clearest way to see it.

    python examples/sql_queries/aggregates_and_having.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    both = bt.sql(
        """
        SELECT l_shipmode,
               COUNT(*) AS lines,
               SUM(l_quantity) AS qty,
               AVG(l_extendedprice) AS avg_price
        FROM lineitem
        WHERE l_quantity > 5
        GROUP BY l_shipmode
        HAVING COUNT(*) > 1000
        ORDER BY qty DESC
        """,
        lineitem=lineitem,
    ).to_pydict()
    print(both["l_shipmode"], both["lines"])

    assert both["qty"] == sorted(both["qty"], reverse=True)
    assert all(value > 1000 for value in both["lines"])

    # The DataFrame spelling: filter, group, aggregate, filter again.
    equivalent = (
        lineitem.filter(col("l_quantity") > 5)
        .group_by("l_shipmode")
        .agg(
            lines=bt.count(),
            qty=col("l_quantity").sum(),
            avg_price=col("l_extendedprice").mean(),
        )
        .filter(col("lines") > 1000)
        .sort("qty", descending=True)
        .to_pydict()
    )
    assert equivalent["l_shipmode"] == both["l_shipmode"]
    assert equivalent["lines"] == both["lines"]

    # WHERE and HAVING are different questions: dropping the row filter changes the
    # counts, and dropping the group filter changes how many groups survive.
    no_where = bt.sql(
        "SELECT l_shipmode, COUNT(*) AS lines FROM lineitem GROUP BY l_shipmode",
        lineitem=lineitem,
    ).to_pydict()
    assert sum(no_where["lines"]) == lineitem.count()
    assert sum(no_where["lines"]) > sum(both["lines"])


if __name__ == "__main__":
    main()
