"""Batcher adapter — the system under test.

Operator-mix cases build on the ``Dataset`` handle (``bt.from_arrow`` /
``bt.read.parquet``); the SQL suites run through a ``bt.Session``, which parses the
query with sqlglot and lowers it to the same plan IR the DataFrame API produces.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt

from .base import Engine, SqlRunner


class BatcherEngine(Engine):
    name = "batcher"
    tier = "both"
    supports_sql = True

    @classmethod
    def available(cls) -> bool:
        return True  # batcher is the package under test; always present

    def handle(self, table: pa.Table) -> bt.Dataset:
        return bt.from_arrow(table)

    def read_parquet(self, uri: str) -> bt.Dataset:
        return bt.read.parquet(uri)

    def sql_runner(self, tables: dict[str, pa.Table]) -> SqlRunner:
        session = bt.Session()
        for name, tbl in tables.items():
            session.register(name, tbl)
        return lambda query: session.sql(query).collect()
