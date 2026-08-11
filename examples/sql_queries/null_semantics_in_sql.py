"""Three-valued logic in SQL, and where it surprises people.

`NULL = NULL` is not true, `NOT IN` over a set containing a null returns nothing, and
`COUNT(column)` differs from `COUNT(*)`. All three are standard, all three are correct, and
all three have cost somebody a day.

    python examples/sql_queries/null_semantics_in_sql.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_custkey")
    lineitem = tpch("lineitem").select("l_orderkey", "l_quantity")

    joined = orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey", how="left")
    nulls = joined.filter(col("l_quantity").is_null()).count()
    print("null quantities:", nulls)
    assert nulls > 0

    counts = bt.sql(
        """
        SELECT
            COUNT(*) AS all_rows,
            COUNT(l_quantity) AS with_quantity,
            SUM(CASE WHEN l_quantity IS NULL THEN 1 ELSE 0 END) AS null_rows
        FROM joined
        """,
        joined=joined,
    ).to_pydict()
    print(counts)

    # COUNT(*) counts rows; COUNT(column) counts non-nulls.
    assert counts["all_rows"][0] - counts["with_quantity"][0] == nulls
    assert counts["null_rows"][0] == nulls

    # A comparison against null is null, so neither branch of an equality keeps them.
    equal = bt.sql(
        "SELECT COUNT(*) AS n FROM joined WHERE l_quantity = 10", joined=joined
    ).to_pydict()["n"][0]
    not_equal = bt.sql(
        "SELECT COUNT(*) AS n FROM joined WHERE l_quantity <> 10", joined=joined
    ).to_pydict()["n"][0]
    print(f"= 10: {equal}, <> 10: {not_equal}, total: {joined.count()}")
    assert equal + not_equal + nulls == joined.count()

    # IS NULL is the only test that finds them.
    found = bt.sql(
        "SELECT COUNT(*) AS n FROM joined WHERE l_quantity IS NULL", joined=joined
    ).to_pydict()["n"][0]
    assert found == nulls

    # And COALESCE is how you decide what they mean.
    filled = bt.sql(
        "SELECT COUNT(*) AS n FROM joined WHERE COALESCE(l_quantity, 0) <> 10",
        joined=joined,
    ).to_pydict()["n"][0]
    assert filled == not_equal + nulls


if __name__ == "__main__":
    main()
