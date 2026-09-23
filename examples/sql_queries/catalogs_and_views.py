"""A directory catalog, SQL over it, and a view that follows its base table.

A `bt.Catalog.from_directory` catalog stores each table as a Delta table under
``<root>/<namespace>/<table>``, so its tables outlive the process. A session attaches it,
``USE`` makes one of its namespaces current, and ``SHOW TABLES`` lists it. A view is stored as
its query text and translated each time it is queried, so it sees rows written to its base
table after the view was created.

Results are compared with DuckDB running the same SQL over the same rows.

    python examples/sql_queries/catalogs_and_views.py
"""

from __future__ import annotations

import tempfile

import duckdb
import pyarrow as pa

import batcher as bt

EVENTS = pa.table({"user_id": [1, 2, 1, 3], "kind": ["view", "buy", "buy", "view"]})
MORE = pa.table({"user_id": [2, 2], "kind": ["buy", "buy"]})
BUYERS = "SELECT user_id, count(*) AS buys FROM events WHERE kind = 'buy' GROUP BY user_id"


def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        warehouse = bt.Catalog.from_directory(root, name="wh")
        warehouse.create_namespace("raw")

        session = bt.Session()
        session.catalog.attach(warehouse)
        session.sql("USE wh.raw")
        position = session.sql("SELECT current_catalog() AS c, current_schema() AS s")
        assert position.to_pydict() == {"c": ["wh"], "s": ["raw"]}

        # Writing a table: the DataFrame spelling, then SQL. After USE an unqualified
        # CREATE TABLE lands in wh.raw, not in the session.
        bt.from_arrow(EVENTS).write.table("wh.raw.events", mode="overwrite", session=session)
        session.sql("CREATE TABLE users AS SELECT DISTINCT user_id FROM events")
        tables = sorted(session.sql("SHOW TABLES").to_pydict()["name"])
        assert tables == ["events", "users"], tables
        assert warehouse.list_tables() == ["raw.events", "raw.users"]

        # A view over the catalog table, answered from the table as it is when queried.
        session.sql(f"CREATE VIEW buyers AS {BUYERS}")
        before = session.sql("SELECT * FROM buyers ORDER BY user_id").to_pydict()
        bt.from_arrow(MORE).write.table("wh.raw.events", mode="append", session=session)
        after = session.sql("SELECT * FROM buyers ORDER BY user_id").to_pydict()
        print("buyers before:", before, "after:", after)

        duck = duckdb.connect()
        duck.register("events", EVENTS)
        expected_before = duck.sql(f"{BUYERS} ORDER BY user_id").to_arrow_table().to_pydict()
        duck.register("events", pa.concat_tables([EVENTS, MORE]))
        expected_after = duck.sql(f"{BUYERS} ORDER BY user_id").to_arrow_table().to_pydict()
        assert before == expected_before, (before, expected_before)
        assert after == expected_after, (after, expected_after)

        # The tables are files, so a new session over the same directory sees them. A view is
        # session state and is not stored there.
        fresh = bt.Session()
        fresh.catalog.attach(bt.Catalog.from_directory(root, name="wh"))
        count = fresh.sql("SELECT count(*) AS n FROM wh.raw.events").to_pydict()
        assert count == {"n": [EVENTS.num_rows + MORE.num_rows]}
        assert "buyers" not in fresh


if __name__ == "__main__":
    main()
