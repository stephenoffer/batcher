"""Reading the two text formats people actually receive data in: CSV and newline-delimited JSON.

The `scan` suite times Parquet layouts, and `scenarios/formats/read.py` times readers outside
the correctness-gated runner. These run inside it: five `lineitem` columns (the two keys, a
quantity, a price and the ship mode, so no date-inference rule decides the answer) written
once per run to a local CSV and a local NDJSON file, then each engine's own reader parses the
file inside the timed call and aggregates by ship mode. The parse is the measurement; the
seven-row result is what the gate compares.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.json as pajson

from registry import EngineQueries, suite

if TYPE_CHECKING:
    from context import Context

textio = suite("ops-frame-textio", dataset="operators")

_COLS = ["l_orderkey", "l_partkey", "l_quantity", "l_extendedprice", "l_shipmode"]


def _written(ctx: Context, kind: str) -> str:
    """`lineitem`'s five columns as one `kind` ("csv" or "ndjson") file, written once."""
    import polars as pl

    directory = tempfile.mkdtemp(prefix="bench-textio-")
    atexit.register(shutil.rmtree, directory, True)
    path = os.path.join(directory, f"lineitem.{kind}")
    frame = pl.from_arrow(ctx.table("lineitem").select(_COLS))
    if kind == "csv":
        frame.write_csv(path)
    else:
        frame.write_ndjson(path)
    return path


def _grouped(t: pa.Table) -> pa.Table:
    out = t.group_by("l_shipmode").aggregate(
        [("l_quantity", "count"), ("l_quantity", "sum"), ("l_extendedprice", "sum")]
    )
    return pa.table(
        {
            "l_shipmode": out["l_shipmode"],
            "n": out["l_quantity_count"],
            "qty": out["l_quantity_sum"],
            "rev": out["l_extendedprice_sum"],
        }
    )


def _engines(ctx: Context, path: str, kind: str) -> EngineQueries:
    names = ctx.names()
    fns: EngineQueries = {}
    if "batcher" in names:
        import batcher as bt

        read = bt.read.csv if kind == "csv" else bt.read.json
        fns["batcher"] = lambda: (
            read(path)
            .group_by("l_shipmode")
            .agg(
                n=bt.col("l_quantity").count(),
                qty=bt.col("l_quantity").sum(),
                rev=bt.col("l_extendedprice").sum(),
            )
            .to_arrow()
        )
    if "duckdb" in names:
        import duckdb

        from engines.duckdb import match_batcher_budget

        con = duckdb.connect()
        match_batcher_budget(con)
        reader = (
            f"read_csv('{path}')"
            if kind == "csv"
            else f"read_json('{path}', format='newline_delimited')"
        )
        sql = (
            "SELECT l_shipmode, count(l_quantity) AS n, sum(l_quantity) AS qty, "
            f"sum(l_extendedprice) AS rev FROM {reader} GROUP BY l_shipmode"
        )
        fns["duckdb"] = lambda: con.sql(sql).to_arrow_table()
    if "polars" in names:
        import polars as pl

        scan = pl.scan_csv if kind == "csv" else pl.scan_ndjson
        fns["polars"] = lambda: (
            scan(path)
            .group_by("l_shipmode")
            .agg(
                pl.col("l_quantity").count().alias("n"),
                pl.col("l_quantity").sum().alias("qty"),
                pl.col("l_extendedprice").sum().alias("rev"),
            )
            .collect()
            .to_arrow()
        )
    if "pyarrow" in names:
        read_pa = pacsv.read_csv if kind == "csv" else pajson.read_json
        fns["pyarrow"] = lambda: _grouped(read_pa(path))
    return fns


@textio.case("op-read-csv")
def read_csv(ctx: Context) -> EngineQueries:
    """Parse a five-column CSV of every `lineitem` row (6M at sf1) and group it by ship mode."""
    return _engines(ctx, _written(ctx, "csv"), "csv")


@textio.case("op-read-ndjson")
def read_ndjson(ctx: Context) -> EngineQueries:
    """Parse the same rows as newline-delimited JSON and group them by ship mode."""
    return _engines(ctx, _written(ctx, "ndjson"), "ndjson")
