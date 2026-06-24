"""SQL over Datasets — and blending SQL with Python.

``bt.sql`` runs a SQL query against one or more Datasets bound by keyword and returns
a lazy Dataset, so SQL and the DataFrame API compose freely. For a reusable catalog
of tables and Python functions, ``bt.Session`` is the DuckDB-connection /
SparkSession analogue. This example covers:

- a plain ``bt.sql`` query (GROUP BY / HAVING / ORDER BY)
- registering a Dataset as a table on a ``Session`` and querying it by name
- calling a registered Python function from SQL (scalar and table forms)
- defining a view with ``CREATE VIEW`` and reading it back
- binding the current Dataset with ``ds.sql("... FROM self")``

    python examples/sql.py
"""

from __future__ import annotations

import pyarrow.compute as pc

import batcher as bt
from batcher import col


def plain_query() -> None:
    """A SQL query bound by keyword; the result keeps composing with the DataFrame API."""
    events = bt.from_pydict(
        {
            "user": ["a", "b", "a", "c", "b"],
            "kind": ["click", "view", "click", "view", "click"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    clicks_by_user = bt.sql(
        """
        SELECT user, SUM(value) AS total
        FROM events
        WHERE kind = 'click'
        GROUP BY user
        HAVING SUM(value) > 1
        ORDER BY total DESC
        """,
        events=events,
    )
    # The SQL result is a Dataset — keep going with the DataFrame API.
    out = clicks_by_user.with_columns(total_rounded=col("total").round(1)).to_pydict()
    print("clicks by user:", out)
    assert out["user"] == ["b", "a"]
    assert out["total"] == [5.0, 4.0]


def session_and_functions() -> None:
    """A Session holds a table catalog and Python functions callable from SQL."""
    session = bt.Session()
    session.register("events", bt.from_pydict({"id": [1, 2, 3, 4], "amount": [10, 20, 30, 40]}))

    # A scalar function — vectorized (it receives an Arrow array). Lowers to the same
    # map_batches path the DataFrame API uses, so Python and SQL share one plan.
    session.register_function("net", lambda a: pc.multiply(a, 0.9))
    scaled = session.sql("SELECT id, net(amount) AS net FROM events WHERE amount >= 20")
    print("scalar UDF:", scaled.to_pydict())
    assert scaled.to_pydict() == {"id": [2, 3, 4], "net": [18.0, 27.0, 36.0]}

    # A table function — transforms a whole relation (declare its output columns).
    def add_band(batch):
        band = pc.if_else(pc.greater(batch.column("amount"), 25), "high", "low")
        return batch.append_column("band", band)

    session.register_function(
        "banded", add_band, table=True, output_columns=["id", "amount", "band"]
    )
    bands = session.sql("SELECT id, band FROM banded(events) ORDER BY id")
    print("table UDF:", bands.to_pydict())
    assert bands.to_pydict() == {"id": [1, 2, 3, 4], "band": ["low", "low", "high", "high"]}

    # CREATE VIEW registers a lazy table in the session; read it back by name.
    session.sql("CREATE VIEW big AS SELECT id, amount FROM events WHERE amount > 25")
    big = session.sql("SELECT id, amount FROM big")
    print("view:", big.to_pydict())
    assert big.to_pydict() == {"id": [3, 4], "amount": [30, 40]}


def dataset_sql() -> None:
    """``ds.sql`` binds the current dataset to a name (``self`` by default)."""
    ds = bt.from_pydict({"x": [1, 2, 3, 4]})
    out = ds.sql("SELECT x, x * x AS sq FROM self WHERE x > 1")
    print("ds.sql:", out.to_pydict())
    assert out.to_pydict() == {"x": [2, 3, 4], "sq": [4, 9, 16]}


def main() -> None:
    plain_query()
    session_and_functions()
    dataset_sql()


if __name__ == "__main__":
    main()
