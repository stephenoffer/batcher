"""DuckDB adapter — the correctness oracle and the primary single-node comparator.

DuckDB reads parquet natively (``read_parquet`` over local/``s3://``/``https://``
paths) and runs every standard-suite query as SQL.
"""

from __future__ import annotations

import importlib.util

import pyarrow as pa

from .base import Engine, SqlRunner


class DuckDBEngine(Engine):
    name = "duckdb"
    tier = "single"
    supports_sql = True

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("duckdb") is not None

    def handle(self, table: pa.Table):
        import duckdb

        con = duckdb.connect()
        con.register("t", table)
        return con  # operator-mix cases query the registered "t"

    def read_parquet(self, uri: str):
        import duckdb

        return duckdb.connect(), uri

    def sql_runner(self, tables: dict[str, pa.Table]) -> SqlRunner:
        import duckdb

        con = duckdb.connect()
        for name, tbl in tables.items():
            con.register(name, tbl)
        return lambda query: con.sql(query).to_arrow_table()
