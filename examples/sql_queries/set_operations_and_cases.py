"""UNION, CASE and IN, written as SQL over real tables.

These are the constructs a ported query uses most, and they all lower onto the same
operators the DataFrame API builds. The check at the end is the one that matters when
porting: the SQL and the DataFrame spelling return the same rows.

Group and order by the alias rather than by an ordinal — `ORDER BY 1` against a computed
projection does not resolve here, and the error names the column it could not find.

    python examples/sql_queries/set_operations_and_cases.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")

    banded = bt.sql(
        """
        SELECT
            CASE
                WHEN o_totalprice < 50000 THEN 'small'
                WHEN o_totalprice < 150000 THEN 'medium'
                ELSE 'large'
            END AS band,
            COUNT(*) AS orders
        FROM orders
        WHERE o_orderpriority IN ('1-URGENT', '2-HIGH')
        GROUP BY band
        ORDER BY band
        """,
        orders=orders,
    ).to_pydict()
    print(banded)

    assert set(banded["band"]) <= {"small", "medium", "large"}
    urgent = orders.filter(col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"])).count()
    assert sum(banded["orders"]) == urgent

    # UNION ALL keeps duplicates; UNION does not.
    both = bt.sql(
        """
        SELECT o_orderkey FROM orders WHERE o_totalprice > 200000
        UNION ALL
        SELECT o_orderkey FROM orders WHERE o_orderstatus = 'F'
        """,
        orders=orders,
    )
    deduped = bt.sql(
        """
        SELECT o_orderkey FROM orders WHERE o_totalprice > 200000
        UNION
        SELECT o_orderkey FROM orders WHERE o_orderstatus = 'F'
        """,
        orders=orders,
    )
    print("union all:", both.count(), "union:", deduped.count())
    assert deduped.count() < both.count()

    # The DataFrame spelling of the same union.
    equivalent = (
        orders.filter(col("o_totalprice") > 200_000)
        .select("o_orderkey")
        .union(orders.filter(col("o_orderstatus") == "F").select("o_orderkey"))
    )
    assert equivalent.count() == both.count()
    assert equivalent.distinct().count() == deduped.count()


if __name__ == "__main__":
    main()
