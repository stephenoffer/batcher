"""SQL over the same engine, and mixing SQL with DataFrame verbs.

``bt.sql`` and ``ds.sql`` build the *same* logical plan the DataFrame API builds, so there
is no second engine and no second semantics. That means you can write the join in SQL and
the feature engineering in expressions, in one pipeline.

    python examples/dataset/sql_interface.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    orders = bt.from_pydict(
        {
            "id": [1, 2, 3, 4],
            "region": ["us", "eu", "us", "eu"],
            "amount": [10, 20, 30, 40],
        }
    )

    # `ds.sql` treats the dataset as `self`.
    totals = orders.sql(
        "SELECT region, SUM(amount) AS total FROM self GROUP BY region ORDER BY region"
    ).to_pydict()
    print(totals)
    assert totals["region"] == ["eu", "us"]
    assert totals["total"] == [60, 40]

    # The same answer through the DataFrame API -- one engine, one plan shape.
    equivalent = orders.group_by("region").agg(total=col("amount").sum()).sort("region").to_pydict()
    assert equivalent == totals

    # A session registers named tables so a query can join them.
    session = bt.Session()
    session.register("orders", orders)
    session.register(
        "regions", bt.from_pydict({"region": ["us", "eu"], "label": ["Americas", "Europe"]})
    )
    joined = session.sql(
        """
        SELECT r.label, SUM(o.amount) AS total
        FROM orders o JOIN regions r ON o.region = r.region
        GROUP BY r.label
        ORDER BY r.label
        """
    ).to_pydict()
    print(joined)
    assert joined["label"] == ["Americas", "Europe"]
    assert joined["total"] == [40, 60]

    # Mixing the two: SQL for the shape, expressions for the derived column.
    mixed = (
        orders.sql("SELECT region, amount FROM self WHERE amount > 15")
        .with_columns(
            bucket=bt.when(col("amount") > 25).then(bt.lit("big")).otherwise(bt.lit("mid"))
        )
        .sort("amount")
        .to_pydict()
    )
    print(mixed)
    assert mixed["amount"] == [20, 30, 40]
    assert mixed["bucket"] == ["mid", "big", "big"]

    # It is still lazy: the SQL is a plan until a terminal call.
    plan = orders.sql("SELECT * FROM self WHERE amount > 15")
    assert plan.count() == 3


if __name__ == "__main__":
    main()
