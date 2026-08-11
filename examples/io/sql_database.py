"""Reading from a SQL database with a connection URI.

The reader takes a connection string and a query and returns a lazy Dataset like any
other source, so the rest of the pipeline cannot tell where the rows came from. SQLite
needs no server, which makes it the one database example that runs anywhere — the URI is
the only thing that changes for Postgres, MySQL or Snowflake.

The driver ships in an optional extra (`pip install 'batcher-engine[sql]'`). This script
says so and exits cleanly when it is absent rather than failing, so the suite still runs
on a machine without it.

    python examples/io/sql_database.py
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    try:
        import adbc_driver_manager.dbapi  # noqa: F401
    except ImportError:
        print("adbc_driver_manager is not installed; install batcher-engine[sql] to run this.")
        return

    nation = tpch("nation").select("n_nationkey", "n_name", "n_regionkey")
    rows = list(nation.to_pandas().itertuples(index=False, name=None))

    with tempfile.TemporaryDirectory() as directory:
        database = str(Path(directory) / "tpch.db")

        connection = sqlite3.connect(database)
        connection.execute("CREATE TABLE nation (n_nationkey INT, n_name TEXT, n_regionkey INT)")
        connection.executemany("INSERT INTO nation VALUES (?, ?, ?)", rows)
        connection.commit()
        connection.close()

        uri = f"sqlite:///{database}"
        loaded = bt.read.sql("SELECT * FROM nation", uri=uri)
        print(loaded.schema)
        assert loaded.count() == 25

        # From here it is an ordinary Dataset.
        summary = (
            loaded.group_by("n_regionkey").agg(nations=bt.count()).sort("n_regionkey").to_pydict()
        )
        print(summary)
        assert sum(summary["nations"]) == 25

        # Pushing the filter into the database is a choice: fewer bytes cross the wire,
        # but the predicate now lives in a string the optimizer cannot see.
        filtered = bt.read.sql("SELECT * FROM nation WHERE n_regionkey = 1", uri=uri)
        assert filtered.count() == loaded.filter(col("n_regionkey") == 1).count()


if __name__ == "__main__":
    main()
