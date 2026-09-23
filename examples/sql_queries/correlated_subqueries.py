"""Correlated subqueries: per-row questions answered as joins, checked against DuckDB.

A correlated subquery asks one question per outer row: "this customer's largest order", "does
this customer have a return". Batcher answers it by decorrelating the subquery into a join on
the correlation key, which is what makes it run in parallel and on a cluster. The shapes below
are the ones where the per-key reading matters most:

- ``ORDER BY … LIMIT 1`` inside the subquery is a top-1 *per customer*, not per table.
- ``EXISTS (… HAVING …)`` asks whether each customer's group passes the ``HAVING``.
- ``count(*) + 1`` for a customer with no orders is 1, because the aggregate still sees an
  empty group.
- A subquery that finds two rows for one customer is an error, as in DuckDB.

Every result is compared with DuckDB running the same SQL over the same tables.

    python examples/sql_queries/correlated_subqueries.py
"""

from __future__ import annotations

import duckdb
import pyarrow as pa

import batcher as bt

CUSTOMERS = pa.table({"cust": [1, 2, 3, 4], "name": ["ana", "bo", "cy", "di"]})
ORDERS = pa.table(
    {
        "order_id": [10, 11, 12, 13, 14, 15],
        "cust": [1, 1, 2, 2, 2, 3],
        "amount": [50, 70, 20, 90, 40, 60],
    }
)

QUERIES = {
    "largest order per customer": """
        SELECT c.name,
               (SELECT o.order_id FROM orders o WHERE o.cust = c.cust
                ORDER BY o.amount DESC LIMIT 1) AS top_order
        FROM customers c ORDER BY c.name""",
    "orders equal to their customer's maximum": """
        SELECT order_id FROM orders o1
        WHERE amount = (SELECT max(amount) FROM orders o2 WHERE o2.cust = o1.cust)
        ORDER BY order_id""",
    "customers with more than two orders": """
        SELECT name FROM customers c
        WHERE EXISTS (SELECT 1 FROM orders o WHERE o.cust = c.cust HAVING count(*) > 2)
        ORDER BY name""",
    "order count plus one, including customers with none": """
        SELECT name, (SELECT count(*) + 1 FROM orders o WHERE o.cust = c.cust) AS n
        FROM customers c ORDER BY name""",
    "spend, zero when there is none": """
        SELECT name, (SELECT coalesce(sum(amount), 0) FROM orders o WHERE o.cust = c.cust)
               AS spend
        FROM customers c ORDER BY name""",
    "has an order over 80, as a column": """
        SELECT name, EXISTS (SELECT 1 FROM orders o WHERE o.cust = c.cust AND o.amount > 80)
               AS big_spender
        FROM customers c ORDER BY name""",
}


def main() -> None:
    session = bt.Session()
    session.register("customers", CUSTOMERS)
    session.register("orders", ORDERS)
    duck = duckdb.connect()
    duck.register("customers", CUSTOMERS)
    duck.register("orders", ORDERS)

    for label, query in QUERIES.items():
        ours = session.sql(query).to_pylist()
        theirs = duck.sql(query).to_arrow_table().to_pylist()
        print(f"{label}: {ours}")
        assert ours == theirs, f"{label}: Batcher {ours} vs DuckDB {theirs}"

    top = session.sql(QUERIES["largest order per customer"]).to_pydict()
    assert top["top_order"] == [11, 13, 15, None]  # di has no orders

    # Two orders for one customer is not a scalar: both engines refuse rather than pick one.
    two_rows = "SELECT name, (SELECT order_id FROM orders o WHERE o.cust = c.cust) FROM customers c"
    try:
        session.sql(two_rows).collect()
    except bt.ExecutionError as exc:
        assert "More than one row" in str(exc)
    else:
        raise AssertionError("a two-row scalar subquery should raise")
    try:
        duck.sql(two_rows).fetchall()
    except duckdb.Error as exc:
        assert "More than one row" in str(exc)
    else:
        raise AssertionError("DuckDB should refuse a two-row scalar subquery too")


if __name__ == "__main__":
    main()
