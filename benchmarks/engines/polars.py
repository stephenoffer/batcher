"""Polars adapter — single-node DataFrame comparator with a SQL surface.

Operator-mix cases build on an eager ``pl.DataFrame``; the standard suites run
through ``pl.SQLContext`` (Polars covers a large SQL subset — queries it cannot
parse surface as ``n/a``/``PARTIAL``, never a wrong answer).
"""

from __future__ import annotations

import importlib.util

import pyarrow as pa

from .base import Engine, SqlRunner


class PolarsEngine(Engine):
    name = "polars"
    tier = "single"
    supports_sql = True

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("polars") is not None

    def handle(self, table: pa.Table):
        import polars as pl

        return pl.from_arrow(table)

    def read_parquet(self, uri: str):
        import polars as pl

        # scan_parquet keeps it lazy; collect happens inside the case.
        return pl.scan_parquet(uri)

    def sql_runner(self, tables: dict[str, pa.Table]) -> SqlRunner:
        import polars as pl

        ctx = pl.SQLContext(eager=True)
        for name, tbl in tables.items():
            ctx.register(name, pl.from_arrow(tbl))
        return lambda query: ctx.execute(query).to_arrow()
