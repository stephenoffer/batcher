"""Querying a file directly from SQL.

A file path bound as a table is the shortest path from "there is a Parquet file" to "here is
the answer". The binding is a Dataset, so everything the DataFrame API can read is available
to a SQL query without an import step.

    python examples/sql_queries/sql_over_files.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch, tpch_path
from batcher import col


def main() -> None:
    # A file on disk, bound straight into a query.
    lineitem = bt.read.parquet(tpch_path("lineitem"))
    result = bt.sql(
        """
        SELECT l_shipmode, COUNT(*) AS lines, SUM(l_quantity) AS qty
        FROM lineitem
        WHERE l_quantity > 20
        GROUP BY l_shipmode
        ORDER BY qty DESC
        """,
        lineitem=lineitem,
    ).to_pydict()
    print(result["l_shipmode"], result["lines"])

    assert result["qty"] == sorted(result["qty"], reverse=True)
    assert sum(result["lines"]) == lineitem.filter(col("l_quantity") > 20).count()

    # A file on S3, the same way.
    from _common import tpch_uri

    remote = bt.read.parquet(tpch_uri("nation"))
    remote_result = bt.sql("SELECT COUNT(*) AS n FROM nation", nation=remote).to_pydict()
    print("nations from S3:", remote_result["n"][0])
    assert remote_result["n"][0] == 25

    # Writing the answer back out, still without leaving SQL for the query itself.
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "summary.parquet")
        bt.sql(
            "SELECT l_shipmode, COUNT(*) AS lines FROM lineitem GROUP BY l_shipmode",
            lineitem=lineitem,
        ).write.parquet(path)

        back = bt.read.parquet(path)
        assert back.count() == lineitem.n_unique("l_shipmode")
        assert sum(back.to_pydict()["lines"]) == lineitem.count()
        print("wrote", back.count(), "rows")

    assert tpch is not None


if __name__ == "__main__":
    main()
